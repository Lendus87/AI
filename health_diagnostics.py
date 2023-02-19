import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array

# Load the pre-trained CNN model
model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')

# Load an image and preprocess it for the model
image = load_img('health_condition.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = tf.keras.applications.vgg16.preprocess_input(image)

# Use the model to make a prediction on the image
predictions = model.predict(image)
predicted_class = tf.keras.applications.vgg16.decode_predictions(predictions, top=1)[0][0]

# Print the results
print('Predicted condition:', predicted_class[1])
