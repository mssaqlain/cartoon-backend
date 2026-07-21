import io
import torch
from flask import Flask, request, send_file
from PIL import Image
 
app = Flask(__name__)
 
# Keep memory usage low on free-tier server
torch.set_num_threads(1)
 
print("Loading AnimeGAN model... (first time takes a bit longer)")
 
# This AI model is trained specifically to turn REAL FACE PHOTOS into
# a fully redrawn anime/cartoon style (not just outlines/filters).
# Model + weights are downloaded automatically from GitHub (free, public).
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device="cpu",
)
face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "face2paint",
    size=512,
)
 
print("Model loaded. Ready to cartoonize.")
 
 
@app.route("/")
def home():
    return "AnimeGAN Cartoon API is running! POST an image to /cartoonize"
 
 
@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    if "image" not in request.files:
        return {"error": "No image uploaded. Send file with key 'image'"}, 400
 
    file = request.files["image"]
 
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}, 400
 
    # Keep input size moderate to save RAM on free server
    img.thumbnail((512, 512))
 
    with torch.no_grad():
        result = face2paint(model, img)
 
    buffer = io.BytesIO()
    result.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
 
    return send_file(buffer, mimetype="image/jpeg")
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
