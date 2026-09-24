"""FastAPI application entry point.

Run:  .venv/bin/uvicorn backend.app.main:app --reload --port 8300
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app import config
from backend.app.db import init_db
from backend.app.routers import datasets, examples, gpu, models, render, stats, training, validate, review, cases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("workbench ready — db=%s adapter=%s", config.DB_PATH, config.ACTIVE_ADAPTER)
    yield


app = FastAPI(
    title="Fine-tuning Workbench",
    version="0.2.0",
    description="Dataset Studio (Phase 1) + Training Launcher (Phase 3).",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"ok": True, "adapter": config.ACTIVE_ADAPTER, "db": str(config.DB_PATH)}


app.include_router(examples.router)
app.include_router(validate.router)
app.include_router(datasets.router)
app.include_router(stats.router)
app.include_router(render.router)
app.include_router(models.router)
app.include_router(review.router)
app.include_router(training.router)
app.include_router(cases.router)
app.include_router(gpu.router)

# UI v1 (T1.5/T1.9): single-page HTML served by the same process, so the
# editor talks to the API same-origin with no build step. CORS only opens
# localhost origins for running the file standalone.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")
