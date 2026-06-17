from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_TARGET = Path("PSL-Flow_GRSL.docx")


TITLE = "PSL-Flow: Physics-Structured Latent Flow Matching for Aerial Visible-to-Infrared Image Translation"

ABSTRACT = (
    "Aerial visible-to-infrared image translation can expand scarce paired VIS-IR data for nighttime "
    "monitoring and remote-sensing perception. Existing methods mainly learn cross-modal mappings in "
    "pixel space or generic latent space. Although recent studies introduce thermal priors, physical "
    "decomposition, or physics-guided losses during training, the inference trajectory is still typically "
    "driven by unstructured image states, which limits explicit modeling of thermal structure formation. "
    "To address this issue, this letter proposes PSL-Flow, a VIS-conditioned flow-matching framework in "
    "a physics-structured latent space for aerial V2IR. The proposed method first builds a TeR-B Net to "
    "estimate temperature proxy, emissivity, environmental radiance, and heat-boundary-aware response from "
    "real infrared images. These factors, together with a residual compensation term, are then encoded by "
    "PSL-VAE into a physics-structured latent state. Finally, SiT is used to learn continuous conditional "
    "transport from Gaussian noise to the target thermal state under visible-image conditioning. Experiments "
    "on AVIID, CART, and DroneVehicle show that PSL-Flow achieves consistently competitive performance, with "
    "the most stable gains appearing in SSIM and LPIPS. In particular, it achieves the best results on all "
    "four metrics on AVIID. Ablation and teacher-derived factor-consistency analysis further indicate that "
    "the proposed latent-state formulation improves thermal structure representation and physics-guided consistency."
)

INDEX_TERMS = (
    "Aerial remote sensing, visible-to-infrared image translation, flow matching, "
    "physics-structured latent space, thermal image generation."
)

INTRO_PARAS = [
    (
        "Infrared imagery is important for aerial remote sensing because it is less sensitive to illumination "
        "changes and better reflects thermal contrast in low-light and complex environments. However, acquiring "
        "well-aligned aerial VIS-IR pairs remains expensive, which limits infrared data collection and downstream "
        "model training. This motivates visible-to-infrared image translation (V2IR) as a practical way to expand "
        "infrared imagery for remote-sensing applications [1]-[4], [15]-[17]."
    ),
    (
        "Unlike generic appearance transfer, aerial V2IR is a physically constrained cross-modal generation problem. "
        "The infrared response is jointly affected by temperature, emissivity, environmental radiation, material "
        "properties, and scene conditions. Existing methods can be roughly grouped into three categories. "
        "GAN- and Transformer-based methods improve global context modeling, structure restoration, and thermal feature "
        "representation [1]-[6], but they still mainly formulate the task as image-style transfer. Diffusion- and "
        "flow-based methods improve distribution transport and sampling stability [7], [8], yet the generation process "
        "is still usually carried out in generic image or latent spaces. Physics-guided methods further inject TeV or "
        "TeR decomposition and physical reconstruction losses into training [9], [10], but physical information is "
        "still mainly used as training-stage supervision rather than an inference-time sampling state."
    ),
    (
        "This leaves a key gap: current methods can make translated results look more infrared-like, but they still do "
        "not explicitly model how thermal structure is formed along the inference trajectory. This is particularly "
        "limiting in aerial scenes with large background variation, weak targets, and ambiguous one-to-many thermal responses."
    ),
    (
        "To address this issue, we propose PSL-Flow, whose core idea is to rewrite aerial V2IR from image generation "
        "to thermal-state generation. As illustrated in Fig. 1, the method first extracts a thermal-response backbone "
        "from real infrared images, then organizes physical factors and residual details into a physics-structured "
        "latent state, and finally performs VIS-conditioned flow matching in that space. The main contributions are "
        "threefold. First, we identify that the main limitation of current physics-guided V2IR methods lies in the "
        "mismatch between training-stage physical supervision and inference-stage unstructured sampling. Second, we "
        "design PSL-Flow, which combines TeR-B Net, PSL-VAE, and SiT to perform conditional transport in a "
        "physics-structured latent space. Third, experiments on three aerial V2IR benchmarks show that this design "
        "yields more stable improvements in structure preservation and perceptual quality, while also providing "
        "supportive evidence of physics-guided consistency."
    ),
]

METHOD_PARAS = [
    (
        "As shown in Fig. 1, PSL-Flow consists of three modules: TeR-B Net, PSL-VAE, and a conditional SiT. "
        "Given a visible image x_vis, the overall inference chain is"
    ),
]

METHOD_A_PARAS = [
    (
        "Instead of directly mapping VIS to IR, we first extract thermal factors from real infrared images. "
        "TeR-B Net predicts temperature proxy T, emissivity e, and environmental radiance R_env, and forms the "
        "thermal backbone"
    ),
    (
        "which serves as a physics-informed approximation of the dominant infrared response. To compensate for details "
        "that are difficult to explain by the backbone alone, we further introduce a heat-boundary-aware response map B, "
        "which emphasizes structure-sensitive regions such as thermal edges and abrupt temperature transitions."
    ),
]

METHOD_B_PARAS = [
    (
        "We define a residual compensation term Δ = x_ir - S_phys and jointly organize [T, e, R_env, B, Δ] as a "
        "structured factor stack. PSL-VAE encodes this stack into a latent thermal state rather than a generic image latent. "
        "During decoding, the final infrared image is reconstructed by combining the decoded thermal backbone and the "
        "boundary-gated residual:"
    ),
    (
        "where G(.) is a boundary-controlled gating function. This design makes the latent representation preserve physical "
        "semantics, thermal backbone structure, and local detail compensation at the same time. The PSL-VAE is trained with "
        "factor reconstruction, physical-backbone consistency, perceptual reconstruction, and KL regularization, while "
        "avoiding the use of a generic image-only compression objective."
    ),
]

METHOD_C_PARAS = [
    (
        "After obtaining the target thermal latent z_phys, we condition SiT on z_vis and learn the continuous transport "
        "from Gaussian noise to the target structured thermal state. Following the standard flow-matching formulation, the "
        "interpolation state and objective are written as"
    ),
    (
        "Unlike prior methods that sample in generic image or latent spaces, PSL-Flow samples in a latent space explicitly "
        "organized by thermal factors. Therefore, the proposed method does not claim strict physical inversion, but it preserves "
        "physical guidance in the inference trajectory through latent-state organization."
    ),
]

EXPERIMENT_PARAS = [
    (
        "Experiments are conducted on AVIID [15], CART [16], and DroneVehicle [17]. All images are resized to 256 x 256. "
        "PSNR, SSIM, LPIPS, and FID are adopted for evaluation. The model is trained on an NVIDIA Tesla A100 80GB GPU "
        "with PyTorch 2.0. AdamW is used with a batch size of 6. Inference is performed with the dopri5 ODE solver using "
        "50 sampling steps. We compare with representative aerial V2IR baselines, including StegoGAN, IRFormer, DR-AVIT, "
        "USTNet, and PID."
    ),
    (
        "The quantitative comparisons are reported in Table I. The most stable improvements of PSL-Flow appear in SSIM and "
        "LPIPS, indicating that the main benefit lies in thermal-structure preservation and perceptual quality rather than "
        "only pixel-level fitting. On AVIID, the proposed method achieves the best results on all four metrics, reaching "
        "23.73/77.04/12.90/30.67 for PSNR/SSIM/LPIPS/FID. On CART, it achieves the best SSIM, LPIPS, and FID, while on "
        "DroneVehicle it gives the best PSNR, SSIM, and LPIPS but not the best FID. This suggests that the current method "
        "is stronger at preserving paired thermal structure than at optimizing global distribution alignment under large "
        "scene variation. The visual comparisons in Fig. 2 are consistent with this observation, showing clearer target "
        "boundaries and more stable thermal layers than the baselines."
    ),
    (
        "Tables II-IV analyze the effect of the structured latent design. Table II shows that replacing a generic KLVAE "
        "with PSL-VAE substantially improves paired reconstruction quality on both AVIID and DroneVehicle, especially in "
        "SSIM and LPIPS, which supports the claim that the proposed latent space is more suitable for thermal-state "
        "representation than generic image compression. Table III further shows that, although the structured latent objective "
        "is harder to optimize at early training stages, it yields better structure-oriented results after sufficient training. "
        "Table IV indicates that the combination of residual compensation and boundary-aware gating produces the most stable "
        "gains in PSNR, SSIM, and LPIPS, confirming that local compensation should be concentrated around thermally sensitive "
        "regions rather than applied uniformly."
    ),
    (
        "Table V reports the complexity comparison. The parameter count changes only marginally, while the additional "
        "calculation mainly comes from structured latent organization and decoding. Despite the higher theoretical FLOPs, "
        "the practical runtime remains acceptable in the current implementation."
    ),
    (
        "We further analyze physical consistency using the same teacher decomposition model adopted in training. Fig. 3 "
        "shows that the generated results remain close to the real infrared images in the teacher-derived factor space. "
        "This evidence is better interpreted as support for physics-guided consistency rather than proof of strict physical "
        "correctness, because the factor analysis is still based on surrogate decomposition rather than direct physical measurements."
    ),
]

CONCLUSION_PARAS = [
    (
        "This letter presented PSL-Flow for aerial visible-to-infrared image translation. The method moves physical guidance "
        "from training-stage supervision to inference-time latent-state sampling by performing VIS-conditioned flow matching "
        "in a physics-structured latent space. Experiments on AVIID, CART, and DroneVehicle show that the proposed design "
        "provides more stable gains in structure preservation and perceptual quality. The current evidence also supports "
        "improved physics-guided consistency, while leaving room for stronger distribution alignment and more direct physical "
        "validation in future work."
    ),
]

EQUATIONS = [
    "z_vis = E_vis(x_vis),    z_phys = Phi_SiT(z_vis, epsilon),    y_hat = D_omega(z_phys)    (1)",
    "S_phys = e ⊙ T + (1 - e) ⊙ R_env    (2)",
    "y_hat = clip(S_hat_phys + G(B_hat) ⊙ Delta_hat, [0,1])    (3)",
    "z_t = alpha_t z_phys + sigma_t epsilon,    v*(z_t,t) = alpha_dot_t z_phys + sigma_dot_t epsilon    (4)",
    "L_flow = E_{z_phys,epsilon,t}[ || v_theta(z_t,t,z_vis) - v*(z_t,t) ||_2^2 ]    (5)",
]

TABLE_I = {
    "caption": "TABLE I\nQuantitative comparison on CART, AVIID, and DroneVehicle.",
    "headers": ["Data", "Method", "PSNR", "SSIM", "LPIPS", "FID"],
    "rows": [
        ["CART", "StegoGAN", "11.98", "36.05", "45.95", "150.55"],
        ["", "IRFormer", "17.12", "54.61", "61.21", "289.87"],
        ["", "DR-AVIT", "14.70", "44.87", "51.28", "218.80"],
        ["", "USTNet", "12.16", "40.01", "46.10", "138.55"],
        ["", "PID", "9.31", "19.90", "39.42", "226.36"],
        ["", "Ours", "14.87", "55.89", "34.72", "125.13"],
        ["AVIID", "StegoGAN", "16.82", "47.30", "28.18", "69.58"],
        ["", "IRFormer", "19.23", "60.12", "44.21", "139.15"],
        ["", "DR-AVIT", "16.52", "46.33", "39.53", "152.93"],
        ["", "USTNet", "18.85", "57.36", "23.53", "54.21"],
        ["", "PID", "22.03", "58.93", "13.30", "52.69"],
        ["", "Ours", "23.73", "77.04", "12.90", "30.67"],
        ["DroneVehicle", "StegoGAN", "13.09", "15.06", "58.80", "149.81"],
        ["", "IRFormer", "15.49", "51.43", "36.65", "112.96"],
        ["", "DR-AVIT", "14.17", "45.91", "27.74", "42.03"],
        ["", "USTNet", "13.55", "45.06", "30.57", "27.52"],
        ["", "PID", "14.03", "34.83", "31.16", "67.91"],
        ["", "Ours", "16.24", "53.58", "23.37", "35.10"],
    ],
}

TABLE_II = {
    "caption": "TABLE II\nReconstruction comparison between KLVAE and PSL-VAE.",
    "headers": ["Data", "Method", "PSNR", "SSIM", "LPIPS", "FID"],
    "rows": [
        ["AVIID", "KLVAE", "32.53", "91.03", "5.83", "17.96"],
        ["", "PSL-VAE", "34.93", "95.37", "5.28", "19.40"],
        ["DroneVehicle", "KLVAE", "25.46", "82.76", "10.15", "15.43"],
        ["", "PSL-VAE", "30.07", "92.09", "5.53", "21.33"],
    ],
}

TABLE_III = {
    "caption": "TABLE III\nGeneration comparison with generic SiT and PSL-Flow at different training steps.",
    "headers": ["Data", "Method", "PSNR", "SSIM", "LPIPS", "FID"],
    "rows": [
        ["AVIID", "SiT (45K)", "24.99", "76.44", "11.87", "29.00"],
        ["", "Ours (45K)", "23.21", "73.39", "15.14", "37.24"],
        ["", "Ours (75K)", "23.73", "77.04", "12.90", "30.60"],
        ["DroneVehicle", "SiT (90K)", "16.51", "51.46", "25.22", "29.90"],
        ["", "Ours (90K)", "16.02", "52.25", "24.44", "37.94"],
        ["", "Ours (120K)", "16.24", "53.58", "23.37", "35.10"],
    ],
}

TABLE_IV = {
    "caption": "TABLE IV\nAblation of residual compensation and boundary-aware gating.",
    "headers": ["Data", "Variant", "PSNR", "SSIM", "LPIPS", "FID"],
    "rows": [
        ["AVIID", "w/o", "23.70", "77.03", "12.92", "31.62"],
        ["", "Delta", "23.68", "76.80", "13.07", "31.50"],
        ["", "B and Delta", "23.73", "77.04", "12.90", "30.67"],
        ["DroneVehicle", "w/o", "16.21", "53.53", "23.38", "34.96"],
        ["", "Delta", "16.22", "53.50", "23.45", "35.03"],
        ["", "B and Delta", "16.24", "53.59", "23.37", "35.10"],
    ],
}

TABLE_V = {
    "caption": "TABLE V\nComplexity comparison between SiT and PSL-Flow.",
    "headers": ["Model", "FLOPs/G", "Params/M", "RT/s"],
    "rows": [
        ["SiT", "14514.9176", "625.1448", "11.916728"],
        ["Ours", "20090.4350", "625.2447", "10.302357"],
    ],
}

REFERENCES = [
    "[1] Z. Han, X. Chen, Z. Ye, et al., \"USTNet: A U-Net Swin Transformer network for aerial visible-to-infrared image translation,\" IEEE Trans. Geosci. Remote Sens., 2025.",
    "[2] D. Ma, J. Su, S. Li, et al., \"AerialIRGAN: Unpaired aerial visible-to-infrared image translation with dual-encoder structure,\" Sci. Rep., vol. 14, no. 1, p. 22105, 2024.",
    "[3] X. Chen, Z. Liu, Z. Han, et al., \"MSFD: Multiscale feature decomposition for cross-modality visible-to-infrared drone image translation,\" IEEE Internet Things J., vol. 12, no. 13, pp. 25951-25965, 2025.",
    "[4] N. Li, H. Wang, H. Zhao, et al., \"Cross-modal visible-to-infrared image translation in remote sensing guided by thermal features,\" IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1-16, 2025.",
    "[5] Z. Han, S. Zhang, Y. Su, et al., \"DR-AVIT: Toward diverse and realistic aerial visible-to-infrared image translation,\" IEEE Trans. Geosci. Remote Sens., vol. 62, pp. 1-13, 2024.",
    "[6] Y. Chen, P. Chen, X. Zhou, et al., \"Implicit multi-spectral transformer: A lightweight and effective visible-to-infrared image translation model,\" in Proc. IJCNN, 2024, pp. 1-8.",
    "[7] X. Wang, W. Cai, Y. Ding, et al., \"RGB to infrared image translation based on diffusion bridges under aerial perspective,\" Remote Sens., vol. 17, no. 22, p. 3703, 2025.",
    "[8] J. Xiao, R. Nayak, N. Zhang, et al., \"ThermalGen: Style-disentangled flow-based generative models for RGB-to-thermal image translation,\" arXiv, 2025.",
    "[9] F. Mao, J. Mei, S. Lu, et al., \"PID: Physics-informed diffusion model for infrared image generation,\" Pattern Recognit., vol. 169, p. 111816, 2026.",
    "[10] H. Yang, M. Tian, J. Wang, et al., \"Realistic infrared image generation based on physics-guided latent diffusion,\" Eng. Appl. Artif. Intell., vol. 167, p. 113752, 2026.",
    "[11] R. Rombach, A. Blattmann, D. Lorenz, et al., \"High-resolution image synthesis with latent diffusion models,\" in Proc. CVPR, 2022, pp. 10684-10695.",
    "[12] N. Ma, M. Goldstein, M. S. Albergo, et al., \"SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers,\" arXiv, 2024.",
    "[13] Q. Zhang, Q. Liu, D. Yuan, et al., \"PPIFuse: Physical priors injected infrared and visible image fusion,\" IEEE Trans. Circuits Syst. Video Technol., 2025.",
    "[14] R. Rombach, A. Blattmann, D. Lorenz, et al., \"High-resolution image synthesis with latent diffusion models,\" in Proc. CVPR, 2022, pp. 10674-10685.",
    "[15] Z. Han, Z. Zhang, S. Zhang, et al., \"Aerial visible-to-infrared image translation: Dataset, evaluation, and baseline,\" J. Remote Sens., vol. 3, p. 0096, 2023.",
    "[16] C. Lee, M. Anderson, N. Raganathan, et al., \"Caltech aerial RGB-thermal dataset in the wild,\" arXiv, 2024.",
    "[17] Y. Sun, B. Cao, P. Zhu, et al., \"Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning,\" arXiv, 2021.",
    "[18] A. Tanchenko, \"Visual-PSNR measure of image quality,\" J. Vis. Commun. Image Represent., vol. 25, no. 5, pp. 874-878, 2014.",
    "[19] Z. Wang, A. C. Bovik, H. R. Sheikh, et al., \"Image quality assessment: From error visibility to structural similarity,\" IEEE Trans. Image Process., vol. 13, no. 4, pp. 600-612, 2004.",
    "[20] R. Zhang, P. Isola, A. A. Efros, et al., \"The unreasonable effectiveness of deep features as a perceptual metric,\" in Proc. CVPR, 2018, pp. 586-595.",
    "[21] M. Heusel, H. Ramsauer, T. Unterthiner, et al., \"GANs trained by a two time-scale update rule converge to a local Nash equilibrium,\" in Proc. NeurIPS, vol. 30, 2017.",
    "[22] S. Wu, Y. Chen, S. Mermet, et al., \"StegoGAN: Leveraging steganography for non-bijective image-to-image translation,\" in Proc. CVPR, 2024, pp. 7922-7931.",
]


def p_text(text: str) -> str:
    return escape(text)


def run(text: str, *, bold: bool = False, italic: bool = False, size: int | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if size is not None:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}<w:t xml:space=\"preserve\">{p_text(text)}</w:t></w:r>"


def paragraph(
    text: str = "",
    *,
    bold: bool = False,
    italic: bool = False,
    size: int | None = None,
    align: str | None = None,
    spacing_before: int | None = None,
    spacing_after: int | None = None,
) -> str:
    ppr = []
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if spacing_before is not None or spacing_after is not None:
        attrs = []
        if spacing_before is not None:
            attrs.append(f'w:before="{spacing_before}"')
        if spacing_after is not None:
            attrs.append(f'w:after="{spacing_after}"')
        ppr.append(f"<w:spacing {' '.join(attrs)}/>")
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    if text:
        return f"<w:p>{ppr_xml}{run(text, bold=bold, italic=italic, size=size)}</w:p>"
    return f"<w:p>{ppr_xml}</w:p>"


def heading(text: str) -> str:
    return paragraph(text, bold=True, size=24, spacing_before=120, spacing_after=80)


def subheading(text: str) -> str:
    return paragraph(text, bold=True, size=22, spacing_before=80, spacing_after=40)


def centered_equation(text: str) -> str:
    return paragraph(text, align="center", spacing_before=40, spacing_after=40)


def figure_placeholder(number: int, caption: str) -> list[str]:
    return [
        paragraph(f"[Insert Fig. {number} here]", italic=True, align="center", spacing_before=80, spacing_after=20),
        paragraph(f"Fig. {number}. {caption}", italic=True, align="center", spacing_after=80),
    ]


def cell(text: str, *, bold: bool = False) -> str:
    jp = "<w:jc w:val=\"center\"/>"
    ppr = f"<w:pPr>{jp}</w:pPr>"
    return (
        "<w:tc>"
        "<w:tcPr><w:tcW w:w=\"2200\" w:type=\"dxa\"/></w:tcPr>"
        f"<w:p>{ppr}{run(text, bold=bold)}</w:p>"
        "</w:tc>"
    )


def table_block(caption: str, headers: list[str], rows: list[list[str]]) -> list[str]:
    blocks = [paragraph(caption, bold=True, align="center", spacing_before=80, spacing_after=40)]
    tbl_pr = (
        "<w:tblPr>"
        "<w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders>"
        "</w:tblPr>"
    )
    header_row = "<w:tr>" + "".join(cell(h, bold=True) for h in headers) + "</w:tr>"
    body_rows = "".join("<w:tr>" + "".join(cell(c) for c in row) + "</w:tr>" for row in rows)
    blocks.append(f"<w:tbl>{tbl_pr}{header_row}{body_rows}</w:tbl>")
    blocks.append(paragraph("", spacing_after=60))
    return blocks


def references_block() -> list[str]:
    blocks = [heading("REFERENCES")]
    blocks.extend(paragraph(ref, spacing_after=20) for ref in REFERENCES)
    return blocks


def extract_sectpr(document_xml: str) -> str:
    match = re.search(r"(<w:sectPr[\s\S]*?</w:sectPr>)", document_xml)
    if match:
        return match.group(1)
    return (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )


def build_document_xml(existing_document_xml: str) -> str:
    blocks: list[str] = []
    blocks.append(paragraph(TITLE, bold=True, size=32, align="center", spacing_after=180))
    blocks.append(paragraph("Abstract— " + ABSTRACT, spacing_after=100))
    blocks.append(paragraph("Index Terms— " + INDEX_TERMS, spacing_after=120))

    blocks.append(heading("I. INTRODUCTION"))
    blocks.extend(paragraph(p, spacing_after=60) for p in INTRO_PARAS)

    blocks.append(heading("II. PROPOSED METHOD"))
    blocks.extend(paragraph(p, spacing_after=60) for p in METHOD_PARAS)
    blocks.append(centered_equation(EQUATIONS[0]))
    blocks.extend(figure_placeholder(1, "Overall framework of PSL-Flow, including TeR-B Net, PSL-VAE, and SiT."))

    blocks.append(subheading("A. Thermal Backbone and Boundary-Aware Decomposition"))
    blocks.append(paragraph(METHOD_A_PARAS[0], spacing_after=60))
    blocks.append(centered_equation(EQUATIONS[1]))
    blocks.append(paragraph(METHOD_A_PARAS[1], spacing_after=60))

    blocks.append(subheading("B. Physics-Structured Latent Modeling"))
    blocks.append(paragraph(METHOD_B_PARAS[0], spacing_after=60))
    blocks.append(centered_equation(EQUATIONS[2]))
    blocks.append(paragraph(METHOD_B_PARAS[1], spacing_after=60))

    blocks.append(subheading("C. Conditional Flow Matching in the Structured Thermal Space"))
    blocks.append(paragraph(METHOD_C_PARAS[0], spacing_after=60))
    blocks.append(centered_equation(EQUATIONS[3]))
    blocks.append(centered_equation(EQUATIONS[4]))
    blocks.append(paragraph(METHOD_C_PARAS[1], spacing_after=60))

    blocks.append(heading("III. EXPERIMENTS"))
    blocks.append(subheading("A. Experimental Settings"))
    blocks.append(paragraph(EXPERIMENT_PARAS[0], spacing_after=60))

    blocks.append(subheading("B. Main Results"))
    blocks.append(paragraph(EXPERIMENT_PARAS[1], spacing_after=60))
    blocks.extend(table_block(TABLE_I["caption"], TABLE_I["headers"], TABLE_I["rows"]))
    blocks.extend(figure_placeholder(2, "Visual comparison on representative samples from AVIID, CART, and DroneVehicle."))

    blocks.append(subheading("C. Ablation and Analysis"))
    blocks.append(paragraph(EXPERIMENT_PARAS[2], spacing_after=60))
    blocks.extend(table_block(TABLE_II["caption"], TABLE_II["headers"], TABLE_II["rows"]))
    blocks.extend(table_block(TABLE_III["caption"], TABLE_III["headers"], TABLE_III["rows"]))
    blocks.extend(table_block(TABLE_IV["caption"], TABLE_IV["headers"], TABLE_IV["rows"]))
    blocks.append(paragraph(EXPERIMENT_PARAS[3], spacing_after=60))
    blocks.extend(table_block(TABLE_V["caption"], TABLE_V["headers"], TABLE_V["rows"]))
    blocks.append(paragraph(EXPERIMENT_PARAS[4], spacing_after=60))
    blocks.extend(figure_placeholder(3, "Teacher-derived physical consistency analysis in the factor space."))

    blocks.append(heading("IV. CONCLUSION"))
    blocks.extend(paragraph(p, spacing_after=60) for p in CONCLUSION_PARAS)

    blocks.extend(references_block())

    sectpr = extract_sectpr(existing_document_xml)
    body = "".join(blocks) + sectpr
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )


def locate_target(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    return DEFAULT_TARGET


def replace_document_xml(docx_path: Path, new_document_xml: str, make_backup: bool) -> None:
    if make_backup:
        backup = docx_path.with_name(docx_path.stem + ".codex-backup.docx")
        shutil.copy2(docx_path, backup)

    with zipfile.ZipFile(docx_path, "r") as zin:
        entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}

    entries["word/document.xml"] = new_document_xml.encode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name, data in entries.items():
                zout.writestr(name, data)
        shutil.copy2(tmp_path, docx_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a GRSL-style PSL-Flow manuscript into a DOCX file.")
    parser.add_argument("--target", help="Path to the target DOCX file.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a sibling backup DOCX.")
    args = parser.parse_args()

    target = locate_target(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Target DOCX not found: {target}")

    with zipfile.ZipFile(target, "r") as zin:
        existing_document_xml = zin.read("word/document.xml").decode("utf-8", errors="ignore")

    new_document_xml = build_document_xml(existing_document_xml)
    replace_document_xml(target, new_document_xml, make_backup=not args.no_backup)
    print(f"Updated: {target}")


if __name__ == "__main__":
    main()
