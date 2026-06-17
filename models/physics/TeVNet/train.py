import argparse
import json
import math
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EvalDataset, TrainDataset
from models import build_tevnet
from utils import AverageMeter, TeVloss, TeVTherNetLoss


def save_json(data, path):
    with open(path, mode="w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def append_jsonl(data, path):
    with open(path, mode="a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=True) + "\n")


def get_model_state_dict(model):
    return (model.module if isinstance(model, nn.DataParallel) else model).state_dict()


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def save_training_checkpoint(
    model,
    path,
    optimizer_obj=None,
    epoch=None,
    best_loss=None,
    best_epoch=None,
    no_improve_validations=None,
    metrics=None,
    args=None,
):
    payload = {
        "format_version": 2,
        "state_dict": get_model_state_dict(model),
    }
    if optimizer_obj is not None:
        payload["optimizer"] = optimizer_obj.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_loss is not None and math.isfinite(float(best_loss)):
        payload["best_loss"] = float(best_loss)
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    if no_improve_validations is not None:
        payload["no_improve_validations"] = int(no_improve_validations)
    if metrics is not None:
        payload["metrics"] = metrics
    if args is not None:
        payload["args"] = vars(args)
    torch.save(payload, path)


def load_checkpoint_flexible(model, ckpt_path, device, optimizer_obj=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value

    target_model = model.module if isinstance(model, nn.DataParallel) else model
    missing, unexpected = target_model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    resume_state = {
        "epoch": None,
        "best_loss": float("inf"),
        "best_epoch": -1,
        "no_improve_validations": 0,
        "optimizer_loaded": False,
    }

    if isinstance(ckpt, dict):
        if ckpt.get("epoch", None) is not None:
            resume_state["epoch"] = int(ckpt["epoch"])
        if ckpt.get("best_loss", None) is not None:
            resume_state["best_loss"] = float(ckpt["best_loss"])
        if ckpt.get("best_epoch", None) is not None:
            resume_state["best_epoch"] = int(ckpt["best_epoch"])
        if ckpt.get("no_improve_validations", None) is not None:
            resume_state["no_improve_validations"] = int(ckpt["no_improve_validations"])
        if optimizer_obj is not None and "optimizer" in ckpt:
            try:
                optimizer_obj.load_state_dict(ckpt["optimizer"])
                resume_state["optimizer_loaded"] = True
            except Exception as exc:
                print(f"[warn] failed to load optimizer state: {exc}")

    return resume_state


def compute_loss(model, lossmodule, inputs, arch):
    if arch in ("thernet", "pp", "phys"):
        preds, aux = model(inputs, return_aux=True)  # type: ignore[arg-type]
        backbone = model.module if isinstance(model, nn.DataParallel) else model
        aux["tim_module"] = getattr(backbone, "tim", None)
        loss = lossmodule(preds, inputs, aux=aux)
    else:
        preds = model(inputs)
        loss = lossmodule.loss_rec(preds, inputs)  # type: ignore[attr-defined]
    return loss


def evaluate_loss(model, loader, lossmodule, device, arch, split_name):
    model.eval()
    meter = AverageMeter()
    start = time.perf_counter()

    with torch.no_grad():
        progress = tqdm(loader, total=len(loader), desc=split_name, leave=False)
        for inputs, _ in progress:
            inputs = inputs.to(device, non_blocking=True)
            loss = compute_loss(model, lossmodule, inputs, arch)
            meter.update(loss.item(), inputs.size(0))
            progress.set_postfix(loss=f"{meter.avg:.6f}")

    return {
        "loss": float(meter.avg),
        "seconds": float(time.perf_counter() - start),
        "num_batches": int(len(loader)),
        "num_samples": int(len(loader.dataset)),
    }


def build_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def main(args):
    os.makedirs(args.outputs_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.outputs_dir, "params.json"))

    history_path = os.path.join(args.outputs_dir, "history.jsonl")
    best_metrics_path = os.path.join(args.outputs_dir, "best_metrics.json")
    last_metrics_path = os.path.join(args.outputs_dir, "last_metrics.json")
    summary_path = os.path.join(args.outputs_dir, "summary.json")
    val_log_path = os.path.join(args.outputs_dir, "val_loss.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_tevnet(args, in_channels=3, out_channels=2 + args.vnums).to(device)
    if torch.cuda.device_count() > 1 and not args.no_dp:
        model = nn.DataParallel(model)

    if args.arch in ("thernet", "pp", "phys"):
        lossmodule = TeVTherNetLoss(
            vnums=args.vnums,
            loss_type=args.loss_type,
            w_rec=args.w_rec,
            w_scene=args.w_scene,
            w_e_smooth=args.w_e_smooth,
            w_mrm_res=args.w_mrm_res,
            w_mrm_assign_tv=args.w_mrm_assign_tv,
            edge_thresh=args.edge_thresh,
        )
    else:
        lossmodule = TeVloss(vnums=args.vnums, loss_type=args.loss_type)

    optimizer_obj = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(img_dir=args.train_dir, image_size=args.image_size)
    if len(train_dataset) == 0:
        raise RuntimeError(f"No train images found under: {args.train_dir}")
    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=bool(args.drop_last_train),
    )

    eval_dataset = EvalDataset(img_dir=args.eval_dir, image_size=args.image_size)
    if len(eval_dataset) == 0:
        raise RuntimeError(f"No validation images found under: {args.eval_dir}")
    eval_loader = build_loader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    test_loader = None
    if args.test_dir:
        test_dataset = EvalDataset(img_dir=args.test_dir, image_size=args.image_size)
        if len(test_dataset) == 0:
            raise RuntimeError(f"No test images found under: {args.test_dir}")
        test_loader = build_loader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    print("[TeVNetTherNet] Run setup")
    print(f"  device={device}")
    print(f"  train_dir={args.train_dir}")
    print(f"  val_dir={args.eval_dir}")
    print(f"  test_dir={args.test_dir or '<disabled>'}")
    print(f"  outputs_dir={args.outputs_dir}")
    print(f"  arch={args.arch}, loss_type={args.loss_type}")
    print(f"  image_size={args.image_size}, train_batch={args.batch_size}, eval_batch={args.eval_batch_size}")
    print(f"  num_workers={args.num_workers}, drop_last_train={bool(args.drop_last_train)}")
    print(f"  train_samples={len(train_dataset)}, val_samples={len(eval_dataset)}")
    if test_loader is not None:
        print(f"  test_samples={len(test_loader.dataset)}")
    print(f"  num_parameters={count_parameters(model):,}")
    print(
        "  early_stop_patience="
        + (str(args.early_stop_patience) if args.early_stop_patience > 0 else "<disabled>")
        + f", early_stop_min_delta={args.early_stop_min_delta}"
    )

    best_loss = float("inf")
    best_epoch = -1
    best_test_loss = None
    no_improve_validations = 0
    start_epoch = int(args.start_epoch)

    if args.resume:
        resume_state = load_checkpoint_flexible(model, args.resume, device, optimizer_obj=optimizer_obj)
        if start_epoch <= 0 and resume_state["epoch"] is not None:
            start_epoch = int(resume_state["epoch"]) + 1
        best_loss = float(resume_state["best_loss"])
        best_epoch = int(resume_state["best_epoch"])
        no_improve_validations = int(resume_state["no_improve_validations"])
        print("[TeVNetTherNet] Resume state")
        print(f"  ckpt={args.resume}")
        print(f"  start_epoch={start_epoch}")
        print(f"  optimizer_loaded={resume_state['optimizer_loaded']}")
        if math.isfinite(best_loss):
            print(f"  best_epoch={best_epoch}, best_loss={best_loss:.8f}")
    elif start_epoch > 0:
        print(f"[TeVNetTherNet] Start from manual epoch={start_epoch} without resume checkpoint.")

    stopped_early = False
    last_epoch = start_epoch - 1
    last_metrics = {}

    for epoch in range(start_epoch, args.num_epochs):
        last_epoch = epoch
        epoch_start = time.perf_counter()
        model.train()
        epoch_losses = AverageMeter()

        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.num_epochs - 1}")
        for inputs, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)

            optimizer_obj.zero_grad(set_to_none=True)
            loss = compute_loss(model, lossmodule, inputs, args.arch)
            loss.backward()
            optimizer_obj.step()

            epoch_losses.update(loss.item(), inputs.size(0))
            progress.set_postfix(
                loss=f"{epoch_losses.avg:.6f}",
                lr=f"{optimizer_obj.param_groups[0]['lr']:.2e}",
            )
            progress.update(1)
        progress.close()

        train_seconds = float(time.perf_counter() - epoch_start)
        metrics = {
            "epoch": int(epoch),
            "train_loss": float(epoch_losses.avg),
            "train_seconds": train_seconds,
            "lr": float(optimizer_obj.param_groups[0]["lr"]),
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss) if math.isfinite(best_loss) else None,
            "no_improve_validations": int(no_improve_validations),
        }

        run_validation = args.num_epochs_val <= 0 or epoch % args.num_epochs_val == 0 or epoch == args.num_epochs - 1
        if run_validation:
            val_metrics = evaluate_loss(
                model=model,
                loader=eval_loader,
                lossmodule=lossmodule,
                device=device,
                arch=args.arch,
                split_name="Validation",
            )
            metrics["val_loss"] = float(val_metrics["loss"])
            metrics["val_seconds"] = float(val_metrics["seconds"])
            print(f"Validation Loss: {val_metrics['loss']:.8f}")

            improved = best_epoch < 0 or val_metrics["loss"] < (best_loss - args.early_stop_min_delta)
            if improved:
                best_loss = float(val_metrics["loss"])
                best_epoch = int(epoch)
                no_improve_validations = 0
                metrics["is_best"] = True

                if test_loader is not None:
                    best_test_metrics = evaluate_loss(
                        model=model,
                        loader=test_loader,
                        lossmodule=lossmodule,
                        device=device,
                        arch=args.arch,
                        split_name="Test(best)",
                    )
                    best_test_loss = float(best_test_metrics["loss"])
                    metrics["best_test_loss"] = best_test_loss
                    metrics["best_test_seconds"] = float(best_test_metrics["seconds"])
                    print(f"Test Loss (best checkpoint): {best_test_loss:.8f}")

                save_training_checkpoint(
                    model=model,
                    path=os.path.join(args.outputs_dir, "best.pth"),
                    optimizer_obj=optimizer_obj,
                    epoch=epoch,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    no_improve_validations=no_improve_validations,
                    metrics=metrics,
                    args=args,
                )
                save_json(metrics, best_metrics_path)
            else:
                no_improve_validations += 1
                metrics["is_best"] = False

            metrics["best_epoch"] = int(best_epoch)
            metrics["best_loss"] = float(best_loss)
            metrics["no_improve_validations"] = int(no_improve_validations)

            with open(val_log_path, "a+", encoding="utf-8") as handle:
                handle.write(
                    "epoch: {epoch}; train_loss: {train_loss:.8f}; val_loss: {val_loss:.8f}; "
                    "best_epoch: {best_epoch}; best_loss: {best_loss:.8f}; no_improve_validations: {no_improve}\n".format(
                        epoch=epoch,
                        train_loss=metrics["train_loss"],
                        val_loss=metrics["val_loss"],
                        best_epoch=best_epoch,
                        best_loss=best_loss,
                        no_improve=no_improve_validations,
                    )
                )

        save_training_checkpoint(
            model=model,
            path=os.path.join(args.outputs_dir, "last.pth"),
            optimizer_obj=optimizer_obj,
            epoch=epoch,
            best_loss=best_loss,
            best_epoch=best_epoch,
            no_improve_validations=no_improve_validations,
            metrics=metrics,
            args=args,
        )
        save_json(metrics, last_metrics_path)
        append_jsonl(metrics, history_path)
        last_metrics = metrics

        if (
            run_validation
            and args.early_stop_patience > 0
            and no_improve_validations >= args.early_stop_patience
        ):
            print(
                f"[TeVNetTherNet] Early stop triggered at epoch {epoch}: "
                f"no improvement for {no_improve_validations} validation rounds."
            )
            stopped_early = True
            break

    summary = {
        "last_epoch": int(last_epoch),
        "stopped_early": bool(stopped_early),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss) if math.isfinite(best_loss) else None,
        "best_test_loss": float(best_test_loss) if best_test_loss is not None else None,
        "last_checkpoint": os.path.join(args.outputs_dir, "last.pth"),
        "best_checkpoint": os.path.join(args.outputs_dir, "best.pth"),
        "history_path": history_path,
    }

    if test_loader is not None and last_epoch >= 0:
        last_test_metrics = evaluate_loss(
            model=model,
            loader=test_loader,
            lossmodule=lossmodule,
            device=device,
            arch=args.arch,
            split_name="Test(last)",
        )
        summary["last_test_loss"] = float(last_test_metrics["loss"])
        summary["last_test_seconds"] = float(last_test_metrics["seconds"])
        print(f"Test Loss (last checkpoint): {last_test_metrics['loss']:.8f}")

    if last_metrics:
        summary["last_train_loss"] = float(last_metrics["train_loss"])
        if "val_loss" in last_metrics:
            summary["last_val_loss"] = float(last_metrics["val_loss"])

    save_json(summary, summary_path)
    print(f"Best epoch: {best_epoch}, Validation loss: {best_loss:.6f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="./data/train", type=str)
    parser.add_argument("--eval-dir", "--val-dir", dest="eval_dir", default="./data/test", type=str)
    parser.add_argument("--test-dir", default="", type=str)
    parser.add_argument("--outputs-dir", default="./experiments/", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-epochs-save", type=int, default=0, help="deprecated; ignored")
    parser.add_argument("--num-epochs-val", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--drop-last-train", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    parser.add_argument("--vnums", type=int, default=4)
    parser.add_argument("--smp_model", type=str, default="PAN")
    parser.add_argument("--smp_encoder", type=str, default="resnet50")
    parser.add_argument("--smp_encoder_weights", type=str, default="imagenet")

    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "thernet", "pp", "phys"])
    parser.add_argument("--loss-type", type=str, default="MSE", choices=["MSE", "L1"])
    parser.add_argument("--no-dp", action="store_true", help="disable DataParallel")

    parser.add_argument("--v-softmax", action="store_true", help="normalize V with softmax (convex env)")
    parser.add_argument("--mrm-k", type=int, default=16)
    parser.add_argument("--mrm-d", type=int, default=32)
    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tim-kernel", type=int, default=5)

    parser.add_argument("--w-rec", type=float, default=1.0)
    parser.add_argument("--w-scene", type=float, default=0.2)
    parser.add_argument("--w-e-smooth", type=float, default=0.05)
    parser.add_argument("--w-mrm-res", type=float, default=0.01)
    parser.add_argument("--w-mrm-assign-tv", type=float, default=0.01)
    parser.add_argument("--edge-thresh", type=float, default=0.08)

    args = parser.parse_args()

    if args.arch in ("thernet", "pp", "phys") and not args.v_softmax:
        args.v_softmax = True

    main(args)
