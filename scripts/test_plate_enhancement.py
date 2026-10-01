from pathlib import Path

import cv2

from app.services.plate_enhancement_service import (
    PlateEnhancementService
)


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    uploads_dir = (
        project_root
        / "data"
        / "uploads"
    )

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    ]:

        image_files.extend(
            uploads_dir.glob(
                extension
            )
        )

    if not image_files:

        print(
            "No images found in "
            "data/uploads/"
        )

        return

    image_path = image_files[-1]

    print(
        f"Testing image: "
        f"{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "Unable to read image."
        )

        return

    # --------------------------------------------------
    # Temporary crop
    # --------------------------------------------------

    height, width = image.shape[:2]

    plate_crop = image[
        int(height * 0.35):
        int(height * 0.65),

        int(width * 0.25):
        int(width * 0.75)
    ]

    # --------------------------------------------------
    # Enhancement
    # --------------------------------------------------

    enhancer = (
        PlateEnhancementService()
    )

    results = enhancer.enhance(
        plate_crop
    )

    output_dir = (
        project_root
        / "data"
        / "processed"
        / "enhancement_test"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Save enhancement variants
    # --------------------------------------------------

    for name, result in results.items():

        output_path = (
            output_dir
            / f"{name}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            result
        )

        print(
            f"Saved: {output_path}"
        )

    print("\nEnhancement complete.")

    print(
        f"\nOutput directory:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()