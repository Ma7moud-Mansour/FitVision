import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.api.v1.endpoints import workout, auth
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")

# API Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(workout.router, prefix="/api/v1", tags=["Workouts"])


@app.on_event("startup")
def on_startup():
    from app.db.models import Base
    from app.db.session import engine
    Base.metadata.create_all(bind=engine)
    logger.info(f"{settings.APP_NAME} is running!")


# Serve frontend pages
@app.get("/", response_class=HTMLResponse)
async def serve_auth():
    with open("app/static/auth.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    with open("app/static/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/upload", response_class=HTMLResponse)
async def serve_upload():
    with open("app/static/upload.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/results/{session_id}", response_class=HTMLResponse)
async def serve_results(session_id: str):
    with open("app/static/results.html", "r", encoding="utf-8") as f:
        return f.read()