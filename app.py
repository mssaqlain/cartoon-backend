import io
import os
import urllib.request
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, send_file

app = Flask(__name__)

MODEL_PATH = "face_paint_512_v2_0.onnx"
MODEL_URL = "https://raw.githubusercontent.com/hpc203/AnimeGAN-onnxruntime/main/face_paint_512_v2_0.onnx"

if not os.path.exists(MODEL_PATH):
    print("Model file not found locally, downloading it (~8.6MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

print("Loading ONNX cartoon model (lightweight, no PyTorch)...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded. Ready to cartoonize.")


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize + normalize image the way the AnimeGAN model expects."""
    img = img.resize((512, 512), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr = arr / 127.5 - 1.0          # scale pixels to [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)   # add batch dimension
    return arr


def postprocess(output: np.ndarray) -> Image.Image:
    """Convert model output back into a normal viewable image."""
    out = output[0]                      # remove batch dim -> CHW
    out = np.transpose(out, (1, 2, 0))   # CHW -> HWC
    out = (out + 1.0) * 127.5            # back to [0, 255]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


@app.route("/")
def home():
    return "AnimeGAN ONNX Cartoon API is running! POST an image to /cartoonize"


@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    if "image" not in request.files:
        return {"error": "No image uploaded. Send file with key 'image'"}, 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}, 400

    input_tensor = preprocess(img)
    outputs = session.run(None, {input_name: input_tensor})
    result_img = postprocess(outputs[0])

    buffer = io.BytesIO()
    result_img.save(buffer, format="JPEG", quality=92)
    buffer.seek(0)

    return send_file(buffer, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
