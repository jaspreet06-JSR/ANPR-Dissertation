from pathlib import Path
from uuid import uuid4
import time

import cv2
from PIL import Image

from app.services.inference_service import (
    InferenceService
)


class ANPRPipeline:
    """
    Main ANPR processing pipeline.

    Current:

        Image
          ↓
        Vehicle Detection
          ↓
        Vehicle Crop
          ↓
        License Plate Detection
          ↓
        Plate Quality Analysis
          ↓
        Annotation
          ↓
        Result

    Future:

        Image Enhancement
          ↓
        OCR
          ↓
        OCR Validation
          ↓
        Multi-frame Fusion
    """

    def __init__(self):

        self.inference_service = (
            InferenceService()
        )

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        self.processed_dir = (
            self.project_root
            / "data"
            / "processed"
        )

        self.processed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def process_image(self, image_path):

        start_time = time.perf_counter()

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: "
                f"{image_path}"
            )

        # --------------------------------------------------
        # LOAD ORIGINAL IMAGE
        # --------------------------------------------------

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        width, height = image.size

        original_image = cv2.imread(
            str(image_path)
        )

        if original_image is None:
            raise ValueError(
                "Unable to read image with OpenCV."
            )

        # This copy is only used for drawing.
        annotated_image = (
            original_image.copy()
        )

        # --------------------------------------------------
        # 1. VEHICLE DETECTION
        # --------------------------------------------------

        vehicles = (
            self.inference_service
            .detect_vehicles(image)
        )

        processed_vehicles = []

        # --------------------------------------------------
        # 2. PROCESS EACH VEHICLE
        # --------------------------------------------------

        for index, vehicle in enumerate(
            vehicles,
            start=1
        ):

            bbox = vehicle["bbox"]

            x1 = bbox["x1"]
            y1 = bbox["y1"]
            x2 = bbox["x2"]
            y2 = bbox["y2"]

            # Keep coordinates inside image
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # --------------------------------------------------
            # VEHICLE ANNOTATION
            # --------------------------------------------------

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            label = (
                f'{vehicle["class_name"]} '
                f'{vehicle["confidence"] * 100:.1f}%'
            )

            text_y = max(
                y1 - 10,
                25
            )

            cv2.putText(
                annotated_image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # --------------------------------------------------
            # VEHICLE CROP
            # --------------------------------------------------

            vehicle_crop = original_image[
                y1:y2,
                x1:x2
            ]

            plate_results = []

            # --------------------------------------------------
            # 3. PLATE DETECTION
            # --------------------------------------------------

            if (
                vehicle_crop is not None
                and vehicle_crop.size > 0
            ):

                plate_detections = (
                    self.inference_service
                    .detect_plates(
                        vehicle_crop
                    )
                )

                # --------------------------------------------------
                # 4. PROCESS EACH PLATE
                # --------------------------------------------------

                for plate in plate_detections:

                    plate_bbox = plate["bbox"]

                    # Coordinates relative to vehicle crop
                    px1 = plate_bbox["x1"]
                    py1 = plate_bbox["y1"]
                    px2 = plate_bbox["x2"]
                    py2 = plate_bbox["y2"]

                    # Keep crop coordinates valid
                    crop_height, crop_width = (
                        vehicle_crop.shape[:2]
                    )

                    px1 = max(
                        0,
                        min(px1, crop_width - 1)
                    )

                    py1 = max(
                        0,
                        min(py1, crop_height - 1)
                    )

                    px2 = max(
                        0,
                        min(px2, crop_width)
                    )

                    py2 = max(
                        0,
                        min(py2, crop_height)
                    )

                    # --------------------------------------------------
                    # CONVERT TO ORIGINAL IMAGE COORDINATES
                    # --------------------------------------------------

                    absolute_x1 = px1 + x1
                    absolute_y1 = py1 + y1
                    absolute_x2 = px2 + x1
                    absolute_y2 = py2 + y1

                    # --------------------------------------------------
                    # PLATE CROP
                    # --------------------------------------------------

                    plate_crop = vehicle_crop[
                        py1:py2,
                        px1:px2
                    ]

                    # --------------------------------------------------
                    # PLATE QUALITY ANALYSIS
                    # --------------------------------------------------

                    quality_result = (
                        self.inference_service
                        .analyze_plate_quality(
                            plate_crop
                        )
                    )

                    # --------------------------------------------------
                    # ENHANCEMENT DECISION
                    # --------------------------------------------------

                    quality_score = (
                        quality_result.get(
                            "quality_score",
                            0
                        )
                    )

                    enhancement_required = (
                        quality_score < 80
                    )

                    enhancement_variants = []

                    if enhancement_required:

                        enhanced_images = (
                            self.inference_service
                            .enhance_plate(
                                plate_crop
                            )
                        )
                        enhancement_variants = list(
                            enhanced_images.keys()
                        )

                    # --------------------------------------------------
                    # PLATE ANNOTATION
                    # --------------------------------------------------

                    cv2.rectangle(
                        annotated_image,
                        (
                            absolute_x1,
                            absolute_y1
                        ),
                        (
                            absolute_x2,
                            absolute_y2
                        ),
                        (255, 0, 0),
                        3
                    )

                    plate_label = (
                        f'PLATE '
                        f'{plate["confidence"] * 100:.1f}%'
                    )

                    cv2.putText(
                        annotated_image,
                        plate_label,
                        (
                            absolute_x1,
                            max(
                                absolute_y1 - 10,
                                25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # --------------------------------------------------
                    # STORE PLATE RESULT
                    # --------------------------------------------------

                    plate_results.append({

                        "confidence":
                            plate["confidence"],

                        "bbox": {

                            "x1":
                                absolute_x1,

                            "y1":
                                absolute_y1,

                            "x2":
                                absolute_x2,

                            "y2":
                                absolute_y2
                        },

                        "quality":
                            quality_result,

                        "enhancement_required":
                            enhancement_required,

                        "enhancement_variants":
                            enhancement_variants,

                        "text":
                            None,

                        "ocr_confidence":
                            None
                    })

            # --------------------------------------------------
            # STORE VEHICLE RESULT
            # --------------------------------------------------

            processed_vehicles.append({

                "vehicle_id":
                    index,

                "vehicle_type":
                    vehicle["class_name"],

                "confidence":
                    vehicle["confidence"],

                "bbox": {

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },

                "license_plates":
                    plate_results
            })

        # --------------------------------------------------
        # 5. SAVE ANNOTATED IMAGE
        # --------------------------------------------------

        result_id = uuid4().hex

        annotated_filename = (
            f"{result_id}.jpg"
        )

        annotated_path = (
            self.processed_dir
            / annotated_filename
        )

        success = cv2.imwrite(
            str(annotated_path),
            annotated_image
        )

        if not success:
            raise RuntimeError(
                "Failed to save annotated image."
            )

        # --------------------------------------------------
        # 6. PROCESSING TIME
        # --------------------------------------------------

        processing_time = (
            time.perf_counter()
            - start_time
        ) * 1000

        # --------------------------------------------------
        # 7. SUMMARY
        # --------------------------------------------------

        plates_detected = sum(
            len(
                vehicle["license_plates"]
            )
            for vehicle
            in processed_vehicles
        )

        result = {

            "result_id":
                result_id,

            "success":
                True,

            "image": {

                "filename":
                    image_path.name,

                "width":
                    width,

                "height":
                    height
            },

            "files": {

                "annotated_image":
                    f"/processed/"
                    f"{annotated_filename}"
            },

            "summary": {

                "vehicles_detected":
                    len(
                        processed_vehicles
                    ),

                "plates_detected":
                    plates_detected,

                "ocr_results":
                    0
            },

            "processing": {

                "time_ms":
                    round(
                        processing_time,
                        2
                    )
            },

            "vehicles":
                processed_vehicles
        }

        return result