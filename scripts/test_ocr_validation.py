from app.services.ocr_validation_service import (
    OCRValidationService
)


def main():

    validator = OCRValidationService(
        minimum_confidence=0.60
    )

    test_cases = [
        {
            "text": "(@KA01MN9787)",
            "confidence": 0.82
        },
        {
            "text": "KA 01 MN 9787",
            "confidence": 0.91
        },
        {
            "text": "dl01ab1234",
            "confidence": 0.88
        },
        {
            "text": "TAX",
            "confidence": 0.3971
        },
        {
            "text": "",
            "confidence": 0.0
        },
        {
            "text": "MH12DE1433",
            "confidence": 0.45
        }
    ]

    print("\nOCR VALIDATION TEST")
    print("=" * 65)

    for index, case in enumerate(
        test_cases,
        start=1
    ):

        result = validator.validate(
            text=case["text"],
            confidence=case["confidence"]
        )

        print(
            f"\nTest {index}"
        )

        print(
            f"Raw text: "
            f"{result['raw_text']!r}"
        )

        print(
            f"Normalized: "
            f"{result['normalized_text']}"
        )

        print(
            f"Format valid: "
            f"{result['format_valid']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence_percent']}%"
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )


if __name__ == "__main__":
    main()