import os
os.environ['TORCH_HOME'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Monkey-patch to handle PosixPath on Windows
import pathlib
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Add YOLOv5 repo path
YOLOV5_DIR = Path(__file__).parent / 'yolov5'
sys.path.insert(0, str(YOLOV5_DIR.resolve()))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes as scale_coords
from utils.torch_utils import select_device

@st.cache_resource
def load_model():
    device = select_device('cpu')
    model = DetectMultiBackend('best.pt', device=device)
    return model, device

model, device = load_model()

st.title("🛄 Airport Security Object Detection")
st.caption("Upload an x-ray image of a bag. The system will detect prohibited items like guns, knives, etc.")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    img_resized, ratio, (dw, dh) = letterbox(image_np, new_shape=640, auto=False)
    img = img_resized.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], image_np.shape).round()

        for *xyxy, conf, cls in pred:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(image_np, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(image_np, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Image", use_column_width=True)
