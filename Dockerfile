FROM python:3.10

WORKDIR /app

COPY . .

# Correction de la commande pour installer les dépendances
RUN pip install -r requirements.txt
RUN pip install --upgrade ultralytics

# Installation de PyTorch avec CUDA 11.8
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install tensorrt
RUN pip install tensorrt_lean
RUN pip install onnx onnxsim onnxruntime-gpu

# Correction de la commande CMD
CMD ["uvicorn", "main:app", "--reload"]
