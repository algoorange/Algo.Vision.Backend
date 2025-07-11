from fastapi import FastAPI
from app.routers import upload, query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
app.include_router(upload.router, prefix="/stream", tags=["streaming"])

@app.get("/")
def home():
    return {"message": "Welcome to the API"}
