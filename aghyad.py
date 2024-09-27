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

# Initialize the model and perform warm-up during application startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set CUDA device and defaults
if device == 'cuda':
    torch.cuda.set_device(0)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_device('cuda')

# Load the YOLO model
model = YOLO('best.pt').to(device)

# Model warm-up with dummy data (640x640 zeros)
dummy_input = torch.zeros(1, 3, 640, 640).to(device)
with torch.no_grad():
    _ = model(dummy_input)
print("Model loaded and warmed up successfully")

def process_image(base64_str):
    """Helper function to process base64-encoded images."""
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 640))
    return img

@app.post('/calculecaptcha')
def calculecaptach(imagebase64: ImageData):
    total_start = datetime.datetime.now()  # Start the total execution timer
    try:
        # Step 1: Image decoding and processing
        step1_start = datetime.datetime.now()
        imagebase64 = imagebase64.image.split(",")[1]  # Decode the base64 image
        img = process_image(imagebase64)  # Preprocess the image

        # # Convert the image to tensor and normalize
        #img_test_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        #img_test_tensor /= 255.0  # Normalize to [0, 1]

        #img_test_tensor = img_test_tensor.to(0)
        #img_test_tensor = img_test_tensor.unsqueeze(0)  # Add batch dimension
        step1_end = datetime.datetime.now()
        print(f"Step 1 (Image Decoding and Processing) Time: {(step1_end - step1_start).total_seconds()} seconds")

        # Step 2: Model inference
        step2_start = datetime.datetime.now()
        with torch.no_grad():
            result = model.predict(img, device=0)  # Inference
        step2_end = datetime.datetime.now()
        print(f"Step 2 (Model Inference) Time: {(step2_end - step2_start).total_seconds()} seconds")

        # Step 3: Extracting and processing bounding boxes
        step3_start = datetime.datetime.now()
        class_list = {0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 11: 'a', 12: 'x'}
        
        boxes_data = result[0].boxes.data.float()  # Extract the bounding boxes

        boxes_list = []
        operator = ''
        
        # Process each detected box
        for box in boxes_data:
            class_id = int(box[5].item())
            class_name = class_list.get(class_id, '')
            if class_name in ['a', '-', 'x']:
                operator = class_name
            else:
                boxes_list.append(int(class_name))

        # Ensure we have two numbers for the operation
        if len(boxes_list) < 2:
            raise ValueError("Not enough numbers detected for the operation.")
        
        step3_end = datetime.datetime.now()
        print(f"Step 3 (Bounding Box Processing) Time: {(step3_end - step3_start).total_seconds()} seconds")

        # Step 4: Calculate the result based on the operator
        step4_start = datetime.datetime.now()
        if operator == 'a':
            myresultcaptch = boxes_list[0] + boxes_list[1]
        elif operator == 'x':
            myresultcaptch = boxes_list[0] * boxes_list[1]
        elif operator == '-':
            myresultcaptch = abs(boxes_list[0] - boxes_list[1])
        else:
            raise ValueError("Operator not recognized.")
        step4_end = datetime.datetime.now()
        print(f"Step 4 (Result Calculation) Time: {(step4_end - step4_start).total_seconds()} seconds")

        total_end = datetime.datetime.now()  # End of total execution time
        print(f"Total Execution Time: {(total_end - total_start).total_seconds()} seconds")

        return {"result": myresultcaptch}

    except Exception as e:
        print(f"Error: {e}")
        return {"result": 0}

    finally:
        print("Done!")