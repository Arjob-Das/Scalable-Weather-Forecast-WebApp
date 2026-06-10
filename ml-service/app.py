import os
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import model

app = FastAPI(title="Weather ML Prediction Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    latitude: float
    longitude: float


@app.post("/api/ml/train")
def train_endpoint(req: TrainRequest):
    try:
        mse, r2 = model.train_model(req.latitude, req.longitude)
        return {
            "status": "success",
            "message": "Model trained successfully",
            "metrics": {
                "mse": mse,
                "r2": r2
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/predict")
def predict_endpoint(
    latitude: float = Query(..., description="Latitude of location"),
    longitude: float = Query(..., description="Longitude of location")
):
    try:
        df_pred = model.make_predictions(latitude, longitude)
        # Convert date to string format for JSON response
        df_pred["date"] = df_pred["date"].dt.strftime("%Y-%m-%d")
        results = df_pred.to_dict(orient="records")
        return {
            "latitude": latitude,
            "longitude": longitude,
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/chart")
def chart_endpoint(
    latitude: float = Query(..., description="Latitude of location"),
    longitude: float = Query(..., description="Longitude of location")
):
    try:
        charts_dir = os.path.join(os.path.dirname(__file__), "charts")
        os.makedirs(charts_dir, exist_ok=True)
        chart_path = os.path.join(
            charts_dir,
            f"forecast_{round(latitude, 2)}_{round(longitude, 2)}.png"
        )

        # Generate forecast chart (internal data validation is handled inside model.py)
        model.generate_forecast_chart(latitude, longitude, chart_path)

        if not os.path.exists(chart_path):
            raise HTTPException(
                status_code=500, detail="Failed to generate chart"
            )

        return FileResponse(chart_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/retrain-new-city")
def retrain_new_city_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Explicitly trigger a full global model retraining to incorporate a new city.
    Retraining runs in the background — the endpoint returns immediately.
    The new city's data will be included in the next run of train_local.py.

    Use GET /api/ml/retrain-status to check whether retraining is in progress.
    """
    try:
        is_known = model._is_known_city(req.latitude, req.longitude)
        model.trigger_new_city_retrain(req.latitude, req.longitude)
        return {
            "status": "accepted",
            "message": (
                "Full retraining started in background for new city."
                if not is_known else
                "City is already in training set; retraining triggered anyway."
            ),
            "latitude": req.latitude,
            "longitude": req.longitude,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/retrain-status")
def retrain_status_endpoint():
    """
    Returns whether a background retraining job is currently running.
    """
    is_running = model._retrain_lock.locked()
    return {
        "retraining_in_progress": is_running,
        "message": (
            "Global model retraining is currently running in the background."
            if is_running else
            "No retraining in progress."
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
