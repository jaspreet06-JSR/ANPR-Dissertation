from pathlib import Path
from importlib import import_module


class VehicleDetector:
    """
    Vehicle detection module using YOLO.

    Detects:
    - Car
    - Motorcycle
    - Bus
    - Truck
    """

    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.25
    ):
        # Project root:
        # ANPR-Dissertation/
        project_root = Path(__file__).resolve().parents[2]

        if model_path is None:
            model_path = project_root / "yolov8n.pt"

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        try:
            yolo = getattr(import_module("ultralytics"), "YOLO")
        except (ImportError, AttributeError) as error:
            raise ImportError(
                "The 'ultralytics' package is required for vehicle detection."
            ) from error

        self.model = yolo(str(self.model_path))

    def detect(self, image):
        """
        Run vehicle detection on an image.

        Returns:
            list of dictionaries containing:
            class name,
            confidence,
            bounding box.
        """

        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Ignore objects that are not vehicles
                if class_id not in self.VEHICLE_CLASSES:
                    continue

                coordinates = box.xyxy[0].tolist()

                x1, y1, x2, y2 = [
                    int(value) for value in coordinates
                ]

                detections.append({
                    "class_id": class_id,
                    "class_name": self.VEHICLE_CLASSES[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        return detections