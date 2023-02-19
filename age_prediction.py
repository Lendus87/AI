import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array

# Load the pre-trained CNN model
model = tf.keras.models.load_model('age_prediction_model.h5')

# Load an image and preprocess it for the model
image = load_img('face.jpg', target_size=(256, 256))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image / 255.0

# Use the model to make a prediction on the image
predicted_age = model.predict(image)[0][0]

# Print the results
print('Predicted age:', int(predicted_age))
