from pathlib import Path

import cv2

from app.services.plate_quality_service import (
    PlateQualityService
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

    # ---------------------------------------------
    # Temporary test crop
    # ---------------------------------------------
    #
    # We don't have a real plate detector yet.
    # Therefore we use a central crop only to
    # test the quality-analysis algorithm.
    #

    height, width = image.shape[:2]

    crop = image[
        int(height * 0.35):
        int(height * 0.65),

        int(width * 0.25):
        int(width * 0.75)
    ]

    analyzer = (
        PlateQualityService()
    )

    result = analyzer.analyze(
        crop
    )

    print("\nPlate Quality Result")
    print("=" * 40)

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Quality Score: "
        f"{result['quality_score']}%"
    )

    print("\nMetrics:")

    for key, value in (
        result["metrics"]
        .items()
    ):

        print(
            f"  {key}: {value}"
        )

    print("\nChecks:")

    for key, value in (
        result["checks"]
        .items()
    ):

        print(
            f"  {key}: {value}"
        )

    print("\nRecommendations:")

    for recommendation in (
        result["recommendations"]
    ):

        print(
            f"  - {recommendation}"
        )


if __name__ == "__main__":
    main()