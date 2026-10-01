import cv2
import numpy as np


class PlateEnhancementService:
    """
    Enhances license plate images before OCR.

    Generates multiple enhancement variants:

    1. Resized image
    2. Denoised image
    3. CLAHE enhanced image
    4. Sharpened image
    5. Adaptive threshold image
    """

    def __init__(self, target_height=120):

        self.target_height = target_height

    def enhance(self, plate_image):
        """
        Generate OCR-friendly versions of a plate image.

        Args:
            plate_image:
                OpenCV BGR image.

        Returns:
            Dictionary containing enhanced images.
        """

        if plate_image is None:

            raise ValueError(
                "Plate image is empty."
            )

        if not isinstance(
            plate_image,
            np.ndarray
        ):

            raise ValueError(
                "Plate image must be a NumPy array."
            )

        if plate_image.size == 0:

            raise ValueError(
                "Plate image contains no pixels."
            )

        # --------------------------------------------------
        # 1. GRAYSCALE
        # --------------------------------------------------

        if len(plate_image.shape) == 3:

            gray = cv2.cvtColor(
                plate_image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = plate_image.copy()

        # --------------------------------------------------
        # 2. RESIZE
        # --------------------------------------------------

        original_height, original_width = (
            gray.shape[:2]
        )

        if original_height <= 0:

            raise ValueError(
                "Invalid plate height."
            )

        scale = (
            self.target_height /
            original_height
        )

        target_width = max(
            1,
            int(original_width * scale)
        )

        resized = cv2.resize(
            gray,
            (
                target_width,
                self.target_height
            ),
            interpolation=cv2.INTER_CUBIC
        )

        # --------------------------------------------------
        # 3. DENOISING
        # --------------------------------------------------

        denoised = cv2.fastNlMeansDenoising(
            resized,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # --------------------------------------------------
        # 4. CLAHE
        # --------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        contrast_enhanced = (
            clahe.apply(denoised)
        )

        # --------------------------------------------------
        # 5. SHARPENING
        # --------------------------------------------------

        sharpening_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        sharpened = cv2.filter2D(
            contrast_enhanced,
            -1,
            sharpening_kernel
        )

        # --------------------------------------------------
        # 6. ADAPTIVE THRESHOLD
        # --------------------------------------------------

        thresholded = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        # --------------------------------------------------
        # RETURN ALL VARIANTS
        # --------------------------------------------------

        return {

            "original_gray":
                gray,

            "resized":
                resized,

            "denoised":
                denoised,

            "contrast_enhanced":
                contrast_enhanced,

            "sharpened":
                sharpened,

            "thresholded":
                thresholded
        }