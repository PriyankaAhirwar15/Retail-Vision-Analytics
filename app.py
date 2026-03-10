from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import io
from PIL import Image

# 1. Create the API 'App'
app = FastAPI()

# 2. Load the 'Brain' again
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

@app.get("/")
def home():
    return {"message": "Welcome to the Retail Vision API! Send an image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 3. Read the image uploaded by the user
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 4. Convert to OpenCV format
    open_cv_image = np.array(image)
    # Convert RGB to BGR (OpenCV uses BGR)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    
    # 5. Run the AI Detection
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 6. Return the count as a JSON response (Professional standard)
    return {
        "status": "success",
        "person_count": len(faces),
        "message": f"I detected {len(faces)} people in the store."
    }

# This is how we run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)