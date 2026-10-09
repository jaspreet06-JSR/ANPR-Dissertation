from pathlib import Path

import cv2

from app.services.plate_visibility_service import (
    PlateVisibilityService
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
            uploads_dir.glob(extension)
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
    # Temporary test crop
    # --------------------------------------------------

    height, width = (
        image.shape[:2]
    )

    crop_x1 = int(
        width * 0.25
    )

    crop_y1 = int(
        height * 0.35
    )

    crop_x2 = int(
        width * 0.75
    )

    crop_y2 = int(
        height * 0.65
    )

    plate_crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    # --------------------------------------------------
    # Analyze
    # --------------------------------------------------

    analyzer = (
        PlateVisibilityService()
    )

    result = analyzer.analyze(
        plate_crop
    )

    print("\nPlate Visibility Result")
    print("=" * 45)

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Visibility Score: "
        f"{result['visibility_score']}%"
    )

    print("\nMetrics:")

    for key, value in (
        result["metrics"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print("\nChecks:")

    for key, value in (
        result["checks"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print("\nWarnings:")

    for warning in result["warnings"]:

        print(
            f"  - {warning}"
        )


if __name__ == "__main__":

    main()