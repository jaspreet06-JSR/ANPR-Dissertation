import cv2
import numpy as np


class PlateQualityService:
    """
    Analyzes the quality of a detected license plate crop.

    Metrics:
    - Blur
    - Brightness
    - Contrast
    - Resolution
    - Aspect ratio

    The service returns a quality score and
    recommendations for the next pipeline stage.
    """

    def __init__(self):

        # Minimum useful plate dimensions
        self.min_width = 60
        self.min_height = 20

        # Blur threshold
        self.blur_threshold = 100.0

        # Brightness range
        self.min_brightness = 50.0
        self.max_brightness = 210.0

        # Contrast threshold
        self.min_contrast = 30.0

    def analyze(self, plate_image):
        """
        Analyze a license plate crop.

        Args:
            plate_image:
                OpenCV BGR image / NumPy array.

        Returns:
            Dictionary containing quality metrics.
        """

        if plate_image is None:
            return {
                "valid": False,
                "quality_score": 0.0,
                "status": "invalid",
                "reason": "Plate image is empty."
            }

        if not isinstance(
            plate_image,
            np.ndarray
        ):
            return {
                "valid": False,
                "quality_score": 0.0,
                "status": "invalid",
                "reason": "Invalid image format."
            }

        if plate_image.size == 0:
            return {
                "valid": False,
                "quality_score": 0.0,
                "status": "invalid",
                "reason": "Plate image contains no pixels."
            }

        height, width = (
            plate_image.shape[:2]
        )

        # ---------------------------------------------
        # Convert to grayscale
        # ---------------------------------------------

        if len(plate_image.shape) == 3:

            gray = cv2.cvtColor(
                plate_image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = plate_image

        # ---------------------------------------------
        # 1. BLUR
        # ---------------------------------------------

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        blur_ok = (
            blur_score >=
            self.blur_threshold
        )

        # ---------------------------------------------
        # 2. BRIGHTNESS
        # ---------------------------------------------

        brightness = float(
            np.mean(gray)
        )

        brightness_ok = (
            self.min_brightness
            <= brightness
            <= self.max_brightness
        )

        # ---------------------------------------------
        # 3. CONTRAST
        # ---------------------------------------------

        contrast = float(
            np.std(gray)
        )

        contrast_ok = (
            contrast >=
            self.min_contrast
        )

        # ---------------------------------------------
        # 4. RESOLUTION
        # ---------------------------------------------

        resolution_ok = (
            width >= self.min_width
            and
            height >= self.min_height
        )

        # ---------------------------------------------
        # 5. ASPECT RATIO
        # ---------------------------------------------

        aspect_ratio = (
            width / height
            if height > 0
            else 0
        )

        # Typical rectangular plate range.
        aspect_ratio_ok = (
            2.0
            <= aspect_ratio
            <= 6.5
        )

        # ---------------------------------------------
        # QUALITY SCORE
        # ---------------------------------------------

        checks = [
            blur_ok,
            brightness_ok,
            contrast_ok,
            resolution_ok,
            aspect_ratio_ok
        ]

        passed_checks = sum(
            checks
        )

        quality_score = (
            passed_checks /
            len(checks)
        ) * 100

        # ---------------------------------------------
        # STATUS
        # ---------------------------------------------

        if quality_score >= 80:

            status = "good"

        elif quality_score >= 60:

            status = "acceptable"

        else:

            status = "poor"

        # ---------------------------------------------
        # RECOMMENDATIONS
        # ---------------------------------------------

        recommendations = []

        if not blur_ok:

            recommendations.append(
                "Image is blurry. "
                "Apply sharpening or deblurring."
            )

        if not brightness_ok:

            if brightness < self.min_brightness:

                recommendations.append(
                    "Plate is too dark. "
                    "Increase brightness."
                )

            else:

                recommendations.append(
                    "Plate is too bright. "
                    "Reduce exposure or brightness."
                )

        if not contrast_ok:

            recommendations.append(
                "Low contrast. "
                "Apply contrast enhancement."
            )

        if not resolution_ok:

            recommendations.append(
                "Plate resolution is too low."
            )

        if not aspect_ratio_ok:

            recommendations.append(
                "Unusual plate aspect ratio. "
                "Check detection crop."
            )

        if not recommendations:

            recommendations.append(
                "Plate quality is suitable for OCR."
            )

        return {

            "valid": True,

            "status":
                status,

            "quality_score":
                round(
                    quality_score,
                    2
                ),

            "metrics": {

                "width":
                    width,

                "height":
                    height,

                "blur":
                    round(
                        blur_score,
                        2
                    ),

                "brightness":
                    round(
                        brightness,
                        2
                    ),

                "contrast":
                    round(
                        contrast,
                        2
                    ),

                "aspect_ratio":
                    round(
                        aspect_ratio,
                        2
                    )
            },

            "checks": {

                "blur":
                    blur_ok,

                "brightness":
                    brightness_ok,

                "contrast":
                    contrast_ok,

                "resolution":
                    resolution_ok,

                "aspect_ratio":
                    aspect_ratio_ok
            },

            "recommendations":
                recommendations
        }