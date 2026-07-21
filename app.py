import cv2
import numpy as np
from flask import Flask, request, send_file
from io import BytesIO
 
app = Flask(__name__)
 
 
def cartoonize_image(img):
    """Converts a normal photo into a cartoon-style image using OpenCV.
    No AI model needed -> 100% free, runs on any free server."""
 
    # Step 1: Resize for consistent processing speed
    img = cv2.resize(img, (0, 0), fx=1, fy=1)
 
    # Step 2: Smooth colors (like a painting)
    color = img
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=200, sigmaSpace=200)
 
    # Step 3: Detect edges (black outlines like cartoon)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2
    )
 
    # Step 4: Combine smooth colors + black outlines
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
 
    return cartoon
 
 
@app.route("/")
def home():
    return "Cartoon API is running! Send a POST request to /cartoonize with an image."
 
 
@app.route("/cartoonize", methods=["POST"])
def cartoonize():
    if "image" not in request.files:
        return {"error": "No image uploaded. Send file with key 'image'"}, 400
 
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
 
    if img is None:
        return {"error": "Invalid image file"}, 400
 
    cartoon = cartoonize_image(img)
 
    # Encode result back to image bytes
    _, buffer = cv2.imencode(".jpg", cartoon)
    result_bytes = BytesIO(buffer)
    result_bytes.seek(0)
 
    return send_file(result_bytes, mimetype="image/jpeg")
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)