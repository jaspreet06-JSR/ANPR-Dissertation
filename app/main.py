from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_health import (
    router as health_router
)

from app.api.routes_detection import (
    router as detection_router
)

from app.api.routes_results import (
    router as results_router
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


FRONTEND_DIR = (
    PROJECT_ROOT /
    "frontend"
)

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)


app = FastAPI(
    title="ANPR Dissertation System",
    description=(
        "Automatic Number Plate "
        "Recognition System"
    ),
    version="1.0.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =====================================================
# API ROUTES
# =====================================================

app.include_router(
    health_router
)

app.include_router(
    detection_router
)

app.include_router(
    results_router
)


# =====================================================
# STATIC FILES
# =====================================================

app.mount(
    "/frontend",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="frontend"
)


app.mount(
    "/processed",
    StaticFiles(
        directory=PROCESSED_DIR
    ),
    name="processed"
)


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():

    return {

        "message":
            "ANPR system is running",

        "version":
            "1.0.0",

        "frontend":
            "/frontend/index.html",

        "docs":
            "/docs"
    }