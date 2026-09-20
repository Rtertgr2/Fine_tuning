"""FastAPI application entry point.

Run:  .venv/bin/uvicorn backend.app.main:app --reload --port 8300
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app import config
from backend.app.db import init_db
from backend.app.routers import datasets, examples, stats, validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("workbench ready — db=%s adapter=%s", config.DB_PATH, config.ACTIVE_ADAPTER)
    yield


app = FastAPI(
    title="Fine-tuning Workbench",
    version="0.1.0",
    description="Dataset Studio (Phase 1): author, validate, version and export fine-tuning data.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"ok": True, "adapter": config.ACTIVE_ADAPTER, "db": str(config.DB_PATH)}


app.include_router(examples.router)
app.include_router(validate.router)
app.include_router(datasets.router)
app.include_router(stats.router)