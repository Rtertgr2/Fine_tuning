#!/bin/bash
# run_server.sh — start uvicorn dev server with auto-reload
cd /run/media/teerametr/3b44566d-03e0-4d76-8469-1bf9cf418a63/Project/Fine-tuning/workbench
.venv/bin/python -m uvicorn backend.app.main:app --port 8300 --reload --log-level info
