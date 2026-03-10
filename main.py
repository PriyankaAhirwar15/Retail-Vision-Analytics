import cv2
import os

def run_retail_detector():
    # 1. Load the 'Brain' (The face detector)
    # This is a built-in rule set from the OpenCV library
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    # 2. Load our Image
    # Make sure you have a photo inside your 'data' folder named 'test_image.jpg'
    image_path = 'data/test_image.jpg'
    
    if not os.path.exists(image_path):
        print(f"Error: I can't find the file at {image_path}. Did you put it in the 'data' folder?")
        return

    img = cv2.imread(image_path)
    
    # 3. Convert to Grayscale (AI sees better in black and white)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Detect Faces (The AI search happens here!)
    # scaleFactor: How much the image size is reduced at each image scale.
    # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 5. Draw boxes and count
    print(f"I found {len(faces)} people in the store!")
    
    for (x, y, w, h) in faces:
        # Draw a Green box (0, 255, 0) around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 6. Save the result
    cv2.imwrite('data/output_result.jpg', img)
    print("Success! I saved the result as 'data/output_result.jpg'. Go check it!")

# This tells Python to run the function above
if __name__ == "__main__":
    run_retail_detector()