from fastapi import APIRouter


router = APIRouter(
    prefix="/api",
    tags=["Health"]
)


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ANPR Backend",
        "message": "ANPR system is running"
    }