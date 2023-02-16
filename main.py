from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from keras.models import load_model

app = FastAPI()


@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):

    # Load the trained h5 model
    model = load_model('trained.h5')

    # Load the input image
    image = cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Check if image is empty
    if image is None:
        return {"Error": "Unable to read input image."}
    else:
        # Preprocess the input image
        image = cv2.resize(image, (300, 300))
        image = image.astype('float') / 255.0
        image = np.expand_dims(image, axis=0)

        # Perform inference
        prediction = model.predict(image)

        # Get the predicted class
        if prediction > 0.5:
            return {"result": "Pneumonia detected"}
        else:
            return {"result": "Pneumonia not detected"}
