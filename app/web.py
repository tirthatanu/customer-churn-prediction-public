
from pathlib import Path

from fastapi.staticfiles import StaticFiles

# Import existing FastAPI application
from app.main import app


# Define project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define frontend directory
FRONTEND_DIR = PROJECT_DIR / "frontend"


# Verify frontend files exist
required_files = [
    "index.html",
    "style.css",
    "script.js"
]

for filename in required_files:
    file_path = FRONTEND_DIR / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required frontend file not found: {file_path}"
        )


# Serve frontend through FastAPI
app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)
