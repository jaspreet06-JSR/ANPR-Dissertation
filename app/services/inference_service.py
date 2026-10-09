from app.models.vehicle_detector import VehicleDetector
from app.models.plate_detector import PlateDetector

from app.services.plate_quality_service import (
    PlateQualityService
)
from app.services.plate_enhancement_service import (
    PlateEnhancementService
)
from app.services.plate_visibility_service import (
    PlateVisibilityService
)

class InferenceService:
    """
    Handles AI inference for the ANPR system.
    """

    def __init__(self):

        self.vehicle_detector = (
            VehicleDetector()
        )

        self.plate_detector = (
            PlateDetector()
        )

        self.plate_quality_service = (
            PlateQualityService()
        )

        self.plate_enhancement_service = (
            PlateEnhancementService()
        )

        self.plate_visibility_service = (
            PlateVisibilityService()
        )

    def detect_vehicles(self, image):

        return (
            self.vehicle_detector
            .detect(image)
        )

    def detect_plates(self, image):

        return (
            self.plate_detector
            .detect(image)
        )

    def analyze_plate_quality(
        self,
        plate_image
    ):

        return (
            self.plate_quality_service
            .analyze(
                plate_image
            )
        )

    def enhance_plate(self, plate_image):

        return (
            self.plate_enhancement_service
            .enhance(plate_image)
        )

    def analyze_plate_visibility(
        self,
        plate_image,
        bbox=None,
        parent_shape=None
    ):

        return (
            self.plate_visibility_service
            .analyze(
                plate_image,
                bbox=bbox,
                parent_shape=parent_shape
            )
        )