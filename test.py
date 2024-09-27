import torch
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
import base64
import datetime
import torchvision

model = YOLO("best.pt")
model.export(format="engine", device=0)

