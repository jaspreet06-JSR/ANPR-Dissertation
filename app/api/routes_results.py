from fastapi import (
    APIRouter,
    HTTPException
)

from app.services.result_service import (
    ResultService
)


router = APIRouter(
    prefix="/api/results",
    tags=["Results"]
)


result_service = ResultService()


@router.get("/")
def get_results():

    results = (
        result_service
        .get_all_results()
    )

    return {
        "count":
            len(results),

        "results":
            results
    }


@router.get("/{result_id}")
def get_result(
    result_id: str
):

    result = (
        result_service
        .get_result(
            result_id
        )
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Result not found."
        )

    return result