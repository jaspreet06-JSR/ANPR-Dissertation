from pathlib import Path

import cv2

from app.models.vehicle_detector import VehicleDetector


def main():

    # Find project root
    project_root = Path(__file__).resolve().parents[1]

    # Input image
    image_path = project_root / "data" / "uploads" / "test1.jpg"

    # Output image
    output_path = (
        project_root
        / "data"
        / "processed"
        / "vehicle_detection.jpg"
    )

    print(f"Input image: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Create detector
    detector = VehicleDetector()

    # Detect vehicles
    detections = detector.detect(image)

    print("\nVehicle detections:")
    print("-" * 50)

    for detection in detections:
        print(
            f"{detection['class_name']:12s} "
            f"confidence={detection['confidence']:.2f} "
            f"bbox={detection['bbox']}"
        )

    print("-" * 50)
    print(f"Total vehicles detected: {len(detections)}")

    # Draw bounding boxes
    output_image = image.copy()

    for detection in detections:

        bbox = detection["bbox"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Save result
    cv2.imwrite(
        str(output_path),
        output_image
    )

    print(f"\nResult saved to:")
    print(output_path)


if __name__ == "__main__":
    main()