from PIL import Image
import torch

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)


class OCRService:
    """
    License plate OCR service using TrOCR.

    Current model:
        microsoft/trocr-base-printed

    Features:
        - MPS acceleration on Apple Silicon
        - OCR text recognition
        - Token-based confidence estimation
        - Multiple image candidate recognition
    """

    def __init__(
        self,
        model_name="microsoft/trocr-base-printed"
    ):

        self.model_name = model_name

        # Use Apple Silicon MPS when available.
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(
            f"Loading TrOCR on {self.device}..."
        )

        # Load processor
        self.processor = (
            TrOCRProcessor
            .from_pretrained(
                self.model_name
            )
        )

        # Load model
        self.model = (
            VisionEncoderDecoderModel
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "TrOCR loaded successfully."
        )

    # ==================================================
    # SINGLE IMAGE OCR
    # ==================================================

    def recognize(
        self,
        plate_image
    ):
        """
        Perform OCR on a single plate image.

        Args:
            plate_image:
                PIL.Image.Image

        Returns:
            Dictionary containing:
                text
                confidence
                confidence_percent
                model
        """

        if plate_image is None:
            raise ValueError(
                "Plate image is empty."
            )

        if not isinstance(
            plate_image,
            Image.Image
        ):
            raise ValueError(
                "plate_image must be a PIL Image."
            )

        # Ensure RGB format
        image = (
            plate_image
            .convert("RGB")
        )

        # ----------------------------------------------
        # Preprocess image
        # ----------------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = (
            inputs.pixel_values
            .to(self.device)
        )

        # ----------------------------------------------
        # Generate OCR sequence
        # ----------------------------------------------

        with torch.no_grad():

            output = self.model.generate(
                pixel_values,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=32
            )

        generated_ids = (
            output.sequences
        )

        # ----------------------------------------------
        # Decode text
        # ----------------------------------------------

        text = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
        )

        text = (
            text
            .strip()
            .upper()
        )

        # ----------------------------------------------
        # Calculate confidence
        # ----------------------------------------------

        confidence = (
            self._calculate_confidence(
                output
            )
        )

        return {
            "text": text,

            "confidence": round(
                confidence,
                4
            ),

            "confidence_percent": round(
                confidence * 100,
                2
            ),

            "model":
                self.model_name
        }

    # ==================================================
    # CONFIDENCE CALCULATION
    # ==================================================

    def _calculate_confidence(
        self,
        output
    ):
        """
        Estimate OCR confidence using the probability
        assigned to each generated token.

        The final confidence is the geometric mean
        of the generated token probabilities.
        """

        scores = output.scores

        if not scores:
            return 0.0

        sequence = (
            output.sequences[0]
        )

        # The first token is normally the decoder
        # start token.
        generated_tokens = (
            sequence[1:]
        )

        token_probabilities = []

        for index, score in enumerate(
            scores
        ):

            if index >= len(
                generated_tokens
            ):
                break

            token_id = int(
                generated_tokens[index]
            )

            # Convert logits to probabilities.
            probabilities = torch.softmax(
                score[0],
                dim=-1
            )

            token_probability = float(
                probabilities[token_id]
            )

            token_probabilities.append(
                token_probability
            )

        if not token_probabilities:
            return 0.0

        # Convert probabilities to a tensor.
        probabilities_tensor = torch.tensor(
            token_probabilities,
            dtype=torch.float32
        )

        # Prevent log(0).
        probabilities_tensor = torch.clamp(
            probabilities_tensor,
            min=1e-10
        )

        # Geometric mean.
        log_probabilities = (
            probabilities_tensor.log()
        )

        confidence = torch.exp(
            log_probabilities.mean()
        )

        return float(
            confidence
        )

    # ==================================================
    # MULTIPLE OCR CANDIDATES
    # ==================================================

    def recognize_candidates(
        self,
        images
    ):
        """
        Run OCR on multiple image variants.

        Args:
            images:
                Dictionary containing PIL images.

                Example:

                {
                    "original": image1,
                    "resized": image2,
                    "sharpened": image3,
                    "thresholded": image4
                }

        Returns:
            List of OCR results sorted by confidence.
        """

        if not isinstance(
            images,
            dict
        ):
            raise ValueError(
                "images must be a dictionary."
            )

        results = []

        for name, image in images.items():

            try:

                result = self.recognize(
                    image
                )

                result["variant"] = name

                results.append(
                    result
                )

            except Exception as error:

                results.append({
                    "variant": name,
                    "text": "",
                    "confidence": 0.0,
                    "confidence_percent": 0.0,
                    "model": self.model_name,
                    "error": str(error)
                })

        # Highest confidence first.
        results.sort(
            key=lambda item:
                item.get(
                    "confidence",
                    0.0
                ),
            reverse=True
        )

        return results

    # ==================================================
    # BEST OCR RESULT
    # ==================================================

    def get_best_result(
        self,
        results
    ):
        """
        Select the highest-confidence OCR result.

        Args:
            results:
                List returned by recognize_candidates().

        Returns:
            Best OCR result.
        """

        if not results:
            return {
                "text": "",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "variant": None,
                "model": self.model_name
            }

        return max(
            results,
            key=lambda item:
                item.get(
                    "confidence",
                    0.0
                )
        )