from fastapi import FastAPI
from app.routers import upload, query, videos, detection, agent
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static frames directory at /frames
app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(videos.router, prefix="/videos", tags=["videos"])
app.include_router(detection.router, prefix="/detection", tags=["detection"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])

@app.get("/")
def home():
    return {"message": "Welcome to the API"}
