"""Script to start the REST API server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "mlops_bobs.api.rest.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )