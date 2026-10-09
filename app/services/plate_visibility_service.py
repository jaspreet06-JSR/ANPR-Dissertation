import cv2
import numpy as np


class PlateVisibilityService:
    """
    Analyzes whether a detected license plate crop
    contains sufficient visible information for OCR.

    This is a heuristic visibility analyzer.

    It does NOT claim to detect every type of physical
    occlusion such as dirt, hands, or objects covering
    characters.
    """

    def __init__(self):

        # Distance from crop boundary used to identify
        # potentially truncated detections.
        self.boundary_margin = 2

        # Minimum useful dimensions.
        self.min_width = 40
        self.min_height = 15

        # Initial edge-density threshold.
        self.min_edge_density = 0.015

    def analyze(
        self,
        plate_image,
        bbox=None,
        parent_shape=None
    ):
        """
        Analyze plate visibility.

        Args:
            plate_image:
                OpenCV BGR or grayscale plate crop.

            bbox:
                Optional plate bounding box relative to
                its parent image/crop.

            parent_shape:
                Optional shape of the parent image.

        Returns:
            Dictionary containing visibility metrics.
        """

        if plate_image is None:

            return {
                "valid": False,
                "visibility_score": 0.0,
                "status": "invalid",
                "reason": "Plate image is empty."
            }

        if not isinstance(
            plate_image,
            np.ndarray
        ):

            return {
                "valid": False,
                "visibility_score": 0.0,
                "status": "invalid",
                "reason": "Invalid image format."
            }

        if plate_image.size == 0:

            return {
                "valid": False,
                "visibility_score": 0.0,
                "status": "invalid",
                "reason": "Plate image contains no pixels."
            }

        # --------------------------------------------------
        # IMAGE DIMENSIONS
        # --------------------------------------------------

        height, width = (
            plate_image.shape[:2]
        )

        # --------------------------------------------------
        # GRAYSCALE
        # --------------------------------------------------

        if len(plate_image.shape) == 3:

            gray = cv2.cvtColor(
                plate_image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = plate_image.copy()

        # --------------------------------------------------
        # 1. RESOLUTION CHECK
        # --------------------------------------------------

        resolution_ok = (
            width >= self.min_width
            and
            height >= self.min_height
        )

        # --------------------------------------------------
        # 2. ASPECT RATIO
        # --------------------------------------------------

        aspect_ratio = (
            width / height
            if height > 0
            else 0
        )

        aspect_ratio_ok = (
            2.0
            <= aspect_ratio
            <= 6.5
        )

        # --------------------------------------------------
        # 3. EDGE DENSITY
        # --------------------------------------------------

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_pixels = np.count_nonzero(
            edges
        )

        total_pixels = (
            width * height
        )

        edge_density = (
            edge_pixels / total_pixels
            if total_pixels > 0
            else 0
        )

        edge_density_ok = (
            edge_density >=
            self.min_edge_density
        )

        # --------------------------------------------------
        # 4. INTENSITY VARIATION
        # --------------------------------------------------

        intensity_std = float(
            np.std(gray)
        )

        intensity_variation_ok = (
            intensity_std >= 15.0
        )

        # --------------------------------------------------
        # 5. BOUNDARY TRUNCATION
        # --------------------------------------------------

        touches_boundary = False

        boundary_details = {
            "left": False,
            "top": False,
            "right": False,
            "bottom": False
        }

        if (
            bbox is not None
            and parent_shape is not None
        ):

            parent_height = (
                parent_shape[0]
            )

            parent_width = (
                parent_shape[1]
            )

            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get(
                "x2",
                parent_width
            )
            y2 = bbox.get(
                "y2",
                parent_height
            )

            margin = (
                self.boundary_margin
            )

            boundary_details["left"] = (
                x1 <= margin
            )

            boundary_details["top"] = (
                y1 <= margin
            )

            boundary_details["right"] = (
                x2 >= parent_width - margin
            )

            boundary_details["bottom"] = (
                y2 >= parent_height - margin
            )

            touches_boundary = any(
                boundary_details.values()
            )

        # --------------------------------------------------
        # 6. VISIBILITY SCORE
        # --------------------------------------------------

        checks = [
            resolution_ok,
            aspect_ratio_ok,
            edge_density_ok,
            intensity_variation_ok,
            not touches_boundary
        ]

        passed_checks = sum(checks)

        visibility_score = (
            passed_checks /
            len(checks)
        ) * 100

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        if visibility_score >= 80:

            status = "good"

        elif visibility_score >= 60:

            status = "partial"

        else:

            status = "poor"

        # --------------------------------------------------
        # WARNINGS
        # --------------------------------------------------

        warnings = []

        if not resolution_ok:

            warnings.append(
                "Plate crop is too small."
            )

        if not aspect_ratio_ok:

            warnings.append(
                "Plate has an unusual aspect ratio."
            )

        if not edge_density_ok:

            warnings.append(
                "Low edge information detected."
            )

        if not intensity_variation_ok:

            warnings.append(
                "Low intensity variation detected."
            )

        if touches_boundary:

            boundaries = [
                name
                for name, value
                in boundary_details.items()
                if value
            ]

            warnings.append(
                "Plate detection touches "
                "the crop boundary: "
                + ", ".join(boundaries)
            )

        if not warnings:

            warnings.append(
                "Plate appears sufficiently "
                "visible for OCR."
            )

        return {

            "valid": True,

            "status":
                status,

            "visibility_score":
                round(
                    visibility_score,
                    2
                ),

            "metrics": {

                "width":
                    width,

                "height":
                    height,

                "aspect_ratio":
                    round(
                        aspect_ratio,
                        2
                    ),

                "edge_density":
                    round(
                        edge_density,
                        4
                    ),

                "intensity_std":
                    round(
                        intensity_std,
                        2
                    ),

                "touches_boundary":
                    touches_boundary
            },

            "checks": {

                "resolution":
                    resolution_ok,

                "aspect_ratio":
                    aspect_ratio_ok,

                "edge_density":
                    edge_density_ok,

                "intensity_variation":
                    intensity_variation_ok,

                "boundary":
                    not touches_boundary
            },

            "boundary":
                boundary_details,

            "warnings":
                warnings
        }