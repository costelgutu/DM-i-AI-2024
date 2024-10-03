# api.py
import uvicorn
from fastapi import FastAPI, HTTPException
import datetime
import time
from model import predict
from loguru import logger
from pydantic import BaseModel
from typing import List
from utils import decode_image

HOST = "0.0.0.0"
PORT = 4321

# Images are loaded via cv2, encoded via base64 and sent as strings
# See utils.py for details
class CellClassificationPredictRequestDto(BaseModel):
    cell: str  # Base64 encoded image string

class CellClassificationPredictResponseDto(BaseModel):
    is_homogenous: int

app = FastAPI()
start_time = time.time()

@app.get('/api')
def hello():
    return {
        "service": "cell-segmentation-usecase",
        "uptime": str(datetime.timedelta(seconds=int(time.time() - start_time)))
    }

@app.get('/')
def index():
    return "Your endpoint is running!"

@app.post('/predict', response_model=CellClassificationPredictResponseDto)
def predict_endpoint(request: CellClassificationPredictRequestDto):
    try:
        # Decode the base64 image
        image = decode_image(request.cell)
        if image is None:
            raise ValueError("Image decoding failed.")

        # Perform prediction
        predicted_homogenous_state = predict(image)

        # Prepare the response
        response = CellClassificationPredictResponseDto(
            is_homogenous=predicted_homogenous_state
        )

        return response

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == '__main__':
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT,
        reload=True  # Optional: Enables auto-reload on code changes
    )
