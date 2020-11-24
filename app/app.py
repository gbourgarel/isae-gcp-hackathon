import base64
import io
import time
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Input(BaseModel):
    model: str
    image: str


class Detection(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_name: str
    confidence: float


class Result(BaseModel):
    detections: List[Detection] = []
    time: float = 0.0
    model: str

# !!!! FILL ME
def parse_predictions(prediction: np.ndarray, classes: [str]) -> List[Detection]:
    detections = []
    for i in range(len(prediction)):        
        box = prediction[i, :4]
        score = float(prediction[i, 4])
        class_idx = int(prediction[i, 5])
        predicted_class = classes[class_idx]
        detections.append(Detection(x_min = box[0], y_min = box[3], x_max = box[2], y_max = box[1], class_name = predicted_class, confidence = score))
    return detections

# !!!! FILL ME
def load_model(model_name: str):
    # Load model from torch
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
    # Evaluation mode + Non maximum threshold
    model = model.fuse().eval()
    # for autoshaping of PIL/cv2/np inputs and NMS
    model = model.autoshape()
    return model


MODEL_NAMES = ["yolov5s", "yolov5m", "yolov5l"]

app = FastAPI(
    title="I'm a fast API, YEAH",
    description="""I'm very very fast API""",
    version="1.0",
)

# !!!! FILL ME
# This is a dictionnary that must contains a model for each key (model names), fill load model
# example: for model_name in MODEL_NAMES: MODELS[model_name] = load_model(model_name)
# You can also lazily load models only when they are called to avoid holding 3 models in memory
MODELS = {}
for m in MODEL_NAMES:
    MODELS[m] = load_model(m)


@app.get("/", description="return the title", response_description="FILL ME", response_model=str)
def root() -> str:
    return app.title


@app.get("/describe", description="FILL ME", response_description="FILL ME", response_model=str)
def describe() -> str:
    return app.description


@app.get("/health", description="FILL ME", response_description="FILL ME", response_model=str)
def health() -> str:
    return "HEALTH OK"


@app.get("/models", description="FILL ME", response_description="FILL ME", response_model=List[str])
def models() -> [str]:
    return MODEL_NAMES


@app.post("/predict", description="FILL ME", response_description="FILL ME", response_model=Result)
def predict(inputs: Input) -> Result:

    # get correct model
    model_name = inputs.model

    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail="wrong model name, choose between {}".format(MODEL_NAMES))

    # Get the model from the list of available models
    model = MODELS.get(model_name)

    # Get & Decode image
    try:
        image = inputs.image.encode("utf-8")
        image = base64.b64decode(image)
        image = Image.open(io.BytesIO(image))
    except:
        raise HTTPException(status_code=400, detail="File is not an image")
    # Convert from RGBA to RGB *to avoid alpha channels*
    if image.mode == "RGBA":
        image = image.convert("RGB")

    # Inference

    # RUN THE PREDICTION, TIME IT
    t0 = time.time()
    predictions = model(image, size=640)  # includes NMS
    t1 = time.time()
    t_res = t1 -t0

    # Post processing
    classes = predictions.names
    predictions = predictions.xyxy[0].numpy()

    # Create a list of [DETECTIONS] objects that match the detection class above, using the parse_predictions method
    detections = parse_predictions(predictions, classes)
    result = Result(detections=detections, time=t_res, model=model_name)

    return result
