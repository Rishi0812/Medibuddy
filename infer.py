import numpy as np
import cv2
from keras.models import load_model

# Load the trained h5 model
model = load_model('trained.h5')

# Load the input image
image = cv2.imread("Data/PNEUMONIA/BACTERIA-40699-0001.jpeg")

# Check if image is empty
if image is None:
    print("Error: Unable to read input image.")
else:
    # Preprocess the input image
    image = cv2.resize(image, (300, 300))
    image = image.astype('float') / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform inference
    prediction = model.predict(image)

    # Get the predicted class
    if prediction > 0.5:
        print('Pneumonia detected')
    else:
        print('Pneumonia not detected')
