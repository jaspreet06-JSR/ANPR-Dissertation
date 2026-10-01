from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from uuid import uuid4

from app.pipeline.anpr_pipeline import (
    ANPRPipeline
)

from app.services.result_service import (
    ResultService
)


router = APIRouter(
    prefix="/api/detection",
    tags=["Detection"]
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

UPLOAD_DIR = (
    PROJECT_ROOT /
    "data" /
    "uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


pipeline = ANPRPipeline()

result_service = ResultService()


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG or WEBP."
            )
        )

    upload_id = uuid4().hex

    saved_filename = (
        f"{upload_id}{extension}"
    )

    file_path = (
        UPLOAD_DIR /
        saved_filename
    )

    try:

        contents = await file.read()

        with open(
            file_path,
            "wb"
        ) as output_file:

            output_file.write(
                contents
            )

        # Run ANPR pipeline

        result = (
            pipeline.process_image(
                file_path
            )
        )

        result["file"] = {

            "original_filename":
                file.filename,

            "saved_filename":
                saved_filename
        }

        # Save result JSON

        result_service.save_result(
            result
        )

        return result

    except Exception as error:

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Image processing failed: "
                f"{str(error)}"
            )
        )