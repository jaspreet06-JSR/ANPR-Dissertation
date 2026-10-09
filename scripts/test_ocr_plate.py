from pathlib import Path

from PIL import Image

from app.services.ocr_service import OCRService


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    plate_dir = (
        project_root
        / "data"
        / "plate_samples"
    )

    # ----------------------------------------------
    # Find plate images
    # ----------------------------------------------

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    ]:

        image_files.extend(
            plate_dir.glob(extension)
        )

    if not image_files:

        print(
            "No plate images found."
        )

        print(
            f"Please place a plate crop inside:"
        )

        print(
            f"{plate_dir}"
        )

        return

    # Use the first available plate crop
    plate_path = image_files[0]

    print(
        f"Testing plate: "
        f"{plate_path.name}"
    )

    # ----------------------------------------------
    # Load plate
    # ----------------------------------------------

    plate_image = (
        Image.open(
            plate_path
        )
        .convert("RGB")
    )

    print(
        f"Plate size: "
        f"{plate_image.width} x "
        f"{plate_image.height}"
    )

    # ----------------------------------------------
    # Create OCR service
    # ----------------------------------------------

    ocr = OCRService()

    # ----------------------------------------------
    # Run OCR
    # ----------------------------------------------

    result = ocr.recognize(
        plate_image
    )

    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    print("\nPlate OCR Result")
    print("=" * 55)

    print(
        f"Text: "
        f"{result['text']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence_percent']}%"
    )

    print(
        f"Model: "
        f"{result['model']}"
    )


if __name__ == "__main__":
    main()