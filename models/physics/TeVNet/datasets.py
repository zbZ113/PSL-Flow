import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_images(img_dir):
    return sorted(
        os.path.join(img_dir, fname)
        for fname in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, fname))
        and os.path.splitext(fname)[1].lower() in VALID_EXTS
    )


def build_train_transforms(image_size):
    return transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size),
        transforms.ToTensor(),
    ])


def build_eval_transforms(image_size):
    return transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

class TrainDataset(Dataset):
    def __init__(self, img_dir, image_size=256, transform=None):
        super(TrainDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform if transform is not None else build_train_transforms(image_size)

        # List all image files in the directories
        self.images = _list_images(img_dir)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load images
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor and normalization
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, image

    def __len__(self):
        return len(self.images)


class EvalDataset(Dataset):
    def __init__(self, img_dir, image_size=256, transform=None):
        super(EvalDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform if transform is not None else build_eval_transforms(image_size)

        # List all image files in the directories
        self.images = _list_images(img_dir)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load images
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor and normalization
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, image

    def __len__(self):
        return len(self.images)
