import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = "best_pneumonia_model.keras"
#IMG_PATH = r"C:\Users\User\Desktop\normal1.jfif"  #96.51   97.96   99.99
#IMG_PATH = r"C:\Users\User\Desktop\pneumonia2bacterial.jfif"  #99.88   70.53   78.90
#IMG_PATH = r"C:\Users\User\Desktop\pneumonia3viral.jfif"  #93.74    63.15   99.98
IMG_PATH = r"C:\Users\User\Desktop\pneumoniafungal.avif"   #93.07   65.02   96.03

IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

img = tf.keras.preprocessing.image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0 

prediction = model.predict(img_array)[0][0]  

label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"\nResult:")
print(f"The picture shows: {label}")
print(f"Probability: {confidence*100:.2f}%")
