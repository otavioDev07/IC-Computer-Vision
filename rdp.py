import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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

    # Fechando os buracos do texto do cupom.
    closed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morphology_kernel)

    # Expande o branco.
    dilated_image = cv2.dilate(closed_image, dilating_kernel)

    contours, _ = cv2.findContours(
        dilated_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError("Contornos não encontrados")

    # Aplicação direta de RDP e filtros simples de área
    img_rows, img_cols = img.shape
    image_area = float(img_rows * img_cols)

    min_area_factor = 0.05
    max_area_factor = 0.98
    border_size = 10
    epsilon_factor = 0.02

    minimum_area = image_area * min_area_factor
    maximum_area = float(
        (img_cols - 2 * border_size) * (img_rows - 2 * border_size)
    ) * max_area_factor
    valid_candidates = []
    rdp_approximations = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = abs(cv2.contourArea(contour))

        if perimeter < 100.0 or area < minimum_area or area >= maximum_area:
            continue

        valid_candidates.append(contour)

        epsilon = perimeter * epsilon_factor
        approximation = cv2.approxPolyDP(contour, epsilon, True)
        rdp_approximations.append(approximation)

        print("-" * 30)
        print("Candidato válido encontrado:")
        print(f"Área: {area:.2f}")
        print(f"Vértices após RDP: {len(approximation)}")
        if len(approximation) > 4:
            print("-> RDP retornou vértices adicionais; possível ocorrência de spikes.")

    if valid_candidates:
        img_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, [valid_candidates[0]], -1, (0, 255, 0), 2)

        img_rdp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_rdp, [rdp_approximations[0]], True, (0, 255, 255), 2)

        for point in rdp_approximations[0]:
            cv2.circle(img_rdp, (point[0][0], point[0][1]), 8, (0, 0, 255), 15)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        stages = [
            (img, "1. Imagem original", "gray"),
            (thresh, "2. Threshold adaptativo", "gray"),
            (closed_image, "3. Fechamento morfológico", "gray"),
            (dilated_image, "4. Dilatação", "gray"),
            (cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB), "5. Contorno entregue ao RDP", None),
            (
                cv2.cvtColor(img_rdp, cv2.COLOR_BGR2RGB),
                f"6. RDP: {len(rdp_approximations[0])} vértices",
                None,
            ),
        ]

        for axis, (stage_image, title, color_map) in zip(axes.flat, stages):
            axis.imshow(stage_image, cmap=color_map)
            axis.set_title(title)
            axis.axis("off")

        fig.suptitle("Pipeline de visualização do RDP", fontsize=18)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhum contorno válido passou pelos critérios de área mínima e máxima.")


if __name__ == "__main__":
    main()
