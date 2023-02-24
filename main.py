from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import glob
from io import BytesIO
import numpy as np
import cv2 as cv
import time
import json


app = FastAPI()

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('yolo/labels.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
# print(class_name)
net = cv.dnn.readNet('yolo/yolov4-tiny.weights', 'yolo/yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))


def detect(img, nn):
    (H, W) = img.shape[:2]
    frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    named_classes = []
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for x in classes:
        named_classes.append(class_name[x])

    return {"classes": named_classes, "scores": scores.tolist(), "boxes": boxes.tolist()}


@app.get("/")
async def root():
    return {"message": "acar object detection endpoint working"}


@app.post("/objectdetection")
def get_body(file: bytes = File(...)):
    image = load_image_into_numpy_array(file)
    results = detect(image, net)
    json_data = json.dumps(results, indent=4)
    return results
