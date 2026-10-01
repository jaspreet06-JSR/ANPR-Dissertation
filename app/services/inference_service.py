from app.models.vehicle_detector import VehicleDetector
from app.models.plate_detector import PlateDetector

from app.services.plate_quality_service import (
    PlateQualityService
)
from app.services.plate_enhancement_service import (
    PlateEnhancementService
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