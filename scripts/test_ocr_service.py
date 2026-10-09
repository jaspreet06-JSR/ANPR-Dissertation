from pathlib import Path

from PIL import Image

from app.services.ocr_service import (
    OCRService
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

    # ----------------------------------------------
    # Find test images
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Load image
    # ----------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # ----------------------------------------------
    # Create OCR service
    # ----------------------------------------------

    ocr = OCRService()

    # ----------------------------------------------
    # Create image variants
    #
    # For now these are simple copies.
    # Actual enhancement variants will be
    # connected later.
    # ----------------------------------------------

    images = {
        "original": image,
        "resized": image.copy(),
        "enhanced": image.copy()
    }

    # ----------------------------------------------
    # Run OCR on all candidates
    # ----------------------------------------------

    results = (
        ocr.recognize_candidates(
            images
        )
    )

    # ----------------------------------------------
    # Display results
    # ----------------------------------------------

    print("\nOCR Candidates")
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nCandidate {index}"
        )

        print(
            f"Variant: "
            f"{result.get('variant', 'unknown')}"
        )

        print(
            f"Text: "
            f"{result.get('text', '')}"
        )

        print(
            f"Confidence: "
            f"{result.get('confidence_percent', 0)}%"
        )

        if result.get("error"):

            print(
                f"Error: "
                f"{result['error']}"
            )

    # ----------------------------------------------
    # Select best result
    # ----------------------------------------------

    best_result = (
        ocr.get_best_result(
            results
        )
    )

    print("\nBest OCR Result")
    print("=" * 60)

    print(
        f"Variant: "
        f"{best_result.get('variant')}"
    )

    print(
        f"Text: "
        f"{best_result.get('text', '')}"
    )

    print(
        f"Confidence: "
        f"{best_result.get('confidence_percent', 0)}%"
    )


if __name__ == "__main__":

    main()