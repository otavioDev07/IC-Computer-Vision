from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera o pipeline gráfico do RDP para todas as imagens de uma pasta."
    )
    parser.add_argument("input_dir", type=Path, help="Pasta com as imagens de entrada.")
    parser.add_argument("output_dir", type=Path, help="Pasta onde as figuras serão salvas.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Também procura imagens nas subpastas da pasta de entrada.",
    )
    return parser.parse_args()


def image_paths(input_dir: Path, output_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    output_dir = output_dir.resolve()
    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and output_dir not in path.resolve().parents
    )


def process_image(image_path: Path, output_path: Path) -> tuple[bool, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem não carregada")

    thresh = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        71,
        2,
    )
    morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilating_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morphology_kernel)
    dilated_image = cv2.dilate(closed_image, dilating_kernel)

    contours, _ = cv2.findContours(
        dilated_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    img_rows, img_cols = img.shape
    image_area = float(img_rows * img_cols)
    minimum_area = image_area * 0.05
    maximum_area = float((img_cols - 20) * (img_rows - 20)) * 0.98
    epsilon_factor = 0.02

    valid_candidates: list[np.ndarray] = []
    rdp_approximations: list[np.ndarray] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = abs(cv2.contourArea(contour))
        if perimeter < 100.0 or area < minimum_area or area >= maximum_area:
            continue

        valid_candidates.append(contour)
        rdp_approximations.append(
            cv2.approxPolyDP(contour, perimeter * epsilon_factor, True)
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    stages: list[tuple[np.ndarray, str, str | None]] = [
        (img, "1. Imagem original", "gray"),
        (thresh, "2. Threshold adaptativo", "gray"),
        (closed_image, "3. Fechamento morfológico", "gray"),
        (dilated_image, "4. Dilatação", "gray"),
    ]

    if valid_candidates:
        img_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, [valid_candidates[0]], -1, (0, 255, 0), 2)

        img_rdp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_rdp, [rdp_approximations[0]], True, (0, 255, 255), 2)
        for point in rdp_approximations[0]:
            cv2.circle(img_rdp, tuple(point[0]), 8, (0, 0, 255), 15)

        stages.extend([
            (cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB), "5. Contorno entregue ao RDP", None),
            (
                cv2.cvtColor(img_rdp, cv2.COLOR_BGR2RGB),
                f"6. RDP: {len(rdp_approximations[0])} vértices",
                None,
            ),
        ])
    else:
        no_candidate = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        stages.extend([
            (no_candidate, "5. Nenhum contorno válido", None),
            (no_candidate, "6. RDP não executado", None),
        ])

    for axis, (stage_image, title, color_map) in zip(axes.flat, stages):
        axis.imshow(stage_image, cmap=color_map)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(f"Pipeline RDP - {image_path.name}", fontsize=18)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return bool(valid_candidates), len(rdp_approximations[0]) if valid_candidates else 0


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise ValueError(f"Pasta de entrada inválida: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = image_paths(args.input_dir, args.output_dir, args.recursive)
    if not paths:
        raise ValueError("Nenhuma imagem encontrada na pasta de entrada")

    print(f"Processando {len(paths)} imagem(ns)...")
    successes = 0
    for index, image_path in enumerate(paths, start=1):
        output_path = args.output_dir / f"{image_path.stem}_pipeline.png"
        try:
            found, vertex_count = process_image(image_path, output_path)
            if found:
                successes += 1
                result = f"{vertex_count} vértices"
            else:
                result = "sem contorno válido"
            print(f"[{index}/{len(paths)}] {image_path.name}: {result}")
        except Exception as error:
            print(f"[{index}/{len(paths)}] {image_path.name}: ERRO - {error}")

    print(f"Concluído: {successes}/{len(paths)} imagens com candidato.\nSaída: {args.output_dir}")


if __name__ == "__main__":
    main()
