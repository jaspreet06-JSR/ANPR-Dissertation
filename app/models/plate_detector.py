from pathlib import Path
from importlib import import_module


class PlateDetector:
    """
    License plate detection module.

    This class is designed to work with a custom
    YOLO license plate detection model.

    The actual trained model can be plugged in later.
    """

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.25
    ):

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        if model_path is None:

            model_path = (
                project_root
                / "models"
                / "plate"
                / "plate.pt"
            )

        self.model_path = Path(
            model_path
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.model = None

        # Model is optional at this stage.
        # This allows the complete application
        # to run before plate training is finished.

        if self.model_path.exists():

            try:

                yolo = getattr(
                    import_module(
                        "ultralytics"
                    ),
                    "YOLO"
                )

                self.model = yolo(
                    str(self.model_path)
                )

            except (
                ImportError,
                AttributeError
            ) as error:

                raise ImportError(
                    "The 'ultralytics' package "
                    "is required for plate detection."
                ) from error

    def detect(self, image):

        """
        Detect license plates.

        Returns:
            list of plate detections.
        """

        # No trained model yet
        if self.model is None:

            return []

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

                confidence = float(
                    box.conf[0]
                )

                coordinates = (
                    box.xyxy[0]
                    .tolist()
                )

                x1, y1, x2, y2 = [
                    int(value)
                    for value in coordinates
                ]

                detections.append({

                    "confidence":
                        round(
                            confidence,
                            4
                        ),

                    "bbox": {

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

        return detections