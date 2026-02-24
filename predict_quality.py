import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "vegetable_model_savedmodel")


print("Loading model...")

model = tf.saved_model.load(MODEL_PATH)

infer = model.signatures["serving_default"]

print("Model loaded successfully")


class_names = [
'bellpepper',
'bitterground',
'capsicum',
'carrot',
'cucumber',
'potato',
'strawberry',
'tomato'
]


def predict(img_path):

    img = image.load_img(img_path, target_size=(224,224))

    img_array = image.img_to_array(img)/255.0

    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    pred = infer(tf.constant(img_array))

    pred = list(pred.values())[0].numpy()

    index = np.argmax(pred)

    confidence = float(pred[0][index])*100

    return class_names[index], confidence


path = input("Enter image path: ")

veg, conf = predict(path)

print("\nRESULT")

print("Vegetable:", veg)

print("Confidence:", round(conf,2), "%")