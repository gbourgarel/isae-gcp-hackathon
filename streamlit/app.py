import requests
import streamlit as st
from PIL import Image
import io
import base64
from pydantic import BaseModel
from typing import List
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import  ImageDraw, ImageFont

# ---- Functions ---


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


@st.cache(show_spinner=True)
def make_dummy_request(model_url: str, model: str, image: Image) -> Result:
    """
    This simulates a fake answer for you to test your application without having access to any other input from other teams
    """
    # We do a dummy encode and decode pass to check that the file is correct
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        buffer: str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data = {"model": model, "image": buffer}

    # We do a dummy decode
    _image = data.get("image")
    _image = _image.encode("utf-8")
    _image = base64.b64decode(_image)
    _image = Image.open(io.BytesIO(_image))  # type: Image
    if _image.mode == "RGBA":
        _image = _image.convert("RGB")

    _model = data.get("model")

    # We generate a random prediction
    w, h = _image.size

    detections = [
        Detection(
            x_min=random.randint(0, w // 2 - 1),
            y_min=random.randint(0, h // 2 - 1),
            x_max=random.randint(w // w, w - 1),
            y_max=random.randint(h // 2, h - 1),
            class_name="dummy",
            confidence=round(random.random(), 3),
        )
        for _ in range(random.randint(1, 10))
    ]

    # We return the result
    result = Result(time=0.1, model=_model, detections=detections)

    return result


@st.cache(show_spinner=True)
def make_request(model_url: str, model: str, image: Image) -> Result:
    """
    Process our data and send a proper request
    """
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        buffer: str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data = {"model": model, "image": buffer}

        response = requests.post("{}/predict".format(model_url), json=data)

    if not response.status_code == 200:
        raise ValueError("Error in processing payload, {}".format(response.text))

    response = response.json()

    return Result.parse_obj(response)


# ---- Streamlit App ---

st.title("NAME ME BECAUSE I AM AWESOME")

with open("APP.md") as f:
    st.markdown(f.read())

# --- Sidebar ---
# defines an h1 header

model_url = st.sidebar.text_input(label="Cluster URL", value="http://localhost:8000")

_model_url = model_url.strip("/")

if st.sidebar.button("Send 'is alive' to IP"):
    try:
        response = requests.get("{}/health".format(_model_url))
        if response.status_code == 200:
            st.sidebar.success("Webapp responding at {}".format(_model_url))
        else:
            st.sidebar.error("Webapp not respond at {}, check url".format(_model_url))
    except ConnectionError:
        st.sidebar.error("Webapp not respond at {}, check url".format(_model_url))

test_mode_on = st.sidebar.checkbox(label="Test Mode - Generate dummy answer", value=False)

# --- Main window

st.markdown("## Inputs")
st.markdown("Describe something... You can also add things like confidence slider etc...")

# Here we should be able to choose between ["yolov5s", "yolov5m", "yolov5l"], perhaps a radio button with the three choices ?
model_name = st.radio("Choose", ['yolov5s', 'yolov5m', 'yolov5l'])

image_file = st.file_uploader("Upload a PNG image", type=([".png", ".jpg"]))

# Converting image, this is done for you :)
if image_file is not None:
    image_file.seek(0)
    image = image_file.read()
    image = Image.open(io.BytesIO(image))
    img_array = np.array(image)

    
def draw_preds(image, result):

    image = image.copy()

#    colors = plt.cm.get_cmap("viridis", len(class_names)).colors
#    colors = (colors[:, :3] * 255.0).astype(np.uint8)
    
    color = (255, 0, 0)

    font = ImageFont.truetype(font="DejaVuSans.ttf", size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
    thickness = (image.size[0] + image.size[1]) // 300

    for detection in result.detections:
        box = [detection.x_min, detection.y_min, detection.x_max, detection.y_max]
        score = float(detection.confidence)
        predicted_class = detection.class_name

        label = "{} {:.2f}".format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype("int32"))
        left = max(0, np.floor(left + 0.5).astype("int32"))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
        right = min(image.size[0], np.floor(right + 0.5).astype("int32"))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for r in range(thickness):
            draw.rectangle([left + r, top + r, right - r, bottom - r], outline=color)
#            draw.rectangle([left + r, top + r, right - r, bottom - r], outline=tuple(colors[class_idx]))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
#        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=tuple(colors[class_idx]))

        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image

def draw_image_with_boxes(image, boxes, header="", description=""):
    # Superpose the semi-transparent object detection boxes.
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += [255, 0, 0]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
#    st.subheader(header)
#    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

if st.button(label="SEND PAYLOAD"):

    if test_mode_on:
        st.warning("Simulating a dummy request to {}".format(model_url))
        result = make_dummy_request(model_url=model_url, model=model_name, image=image)
    else:
        result = make_request(model_url=model_url, model=model_name, image=image)
        print(result.detections)

    st.balloons()

    st.markdown("## Display")

    st.markdown("Make something pretty, draw polygons and confidence..., here's an ugly output")
    
#    boxes = []
#    for d in result.detections:
#        boxes.append([d.x_min, d.y_min, d.x_max, d.y_max])
    image = draw_preds(image, result)
#    draw_image_with_boxes(img_array, boxes=boxes)#, header=d['class_name'], description=d['confidence'])

    st.image(image, width=512, caption="Uploaded Image")

    st.text("Model : {}".format(result.model))
    st.text("Processing time : {}s".format(result.time))

    for detection in result.detections:
        st.json(detection.json())
        
