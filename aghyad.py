from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import base64
import datetime
import torch
import einops
import time
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "https://www.ecsc.gov.sy",
    "https://ecsc.gov.sy",
    "https://ecsc.gov.sy/requests",
    "https://www.ecsc.gov.sy/requests/*",
    "https://www.ecsc.gov.sy/requests",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
model = YOLO('best yolov10.pt').to(device)

# Pre-compile the model
img_test = np.zeros((225, 225, 3), dtype=np.uint8)
_ = model.predict(img_test, device=device)
model.fuse()  # Fuse Conv2d + BatchNorm2d layers
model.half()  # Use half-precision (FP16)

def process_image(base64_str):
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = cv2.resize(img, (320, 320))
    img = img / 255
    return img

class_list = {0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'a', 12: 'x'}

def calculate_captcha(imagebase64):
    l = []
    operator = ''
    img = process_image(imagebase64)

    with torch.no_grad():
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device).half()
        start_time = time.time()
        result = model.predict(img_tensor, device=device)
        inference_time = time.time() - start_time
        print(f"Inference Time: {inference_time * 1000:.2f} ms")  # Convert to milliseconds

    a = result[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        class_name = class_list[d]

        if class_name in ['a', '-', 'x']:
            operator = class_name
        else:
            nombre = int(class_name)
            l.append(nombre)

    myresultcaptch = 0
    if operator == 'a':
        myresultcaptch = l[0] + l[1]
    elif operator == 'x':
        myresultcaptch = l[0] * l[1]
    elif operator == '-':
        myresultcaptch = abs(l[0] - l[1])

    return myresultcaptch

executor = ThreadPoolExecutor(max_workers=6)

@app.post('/calculecaptcha')
def calculecaptach(imagebase64: ImageData):
    imagebase64_str = imagebase64.image.split(",")[1]
    
    future = executor.submit(calculate_captcha, imagebase64_str)
    myresultcaptch = future.result()
    
    return {"result": myresultcaptch}
