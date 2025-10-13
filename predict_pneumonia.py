import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = "best_pneumonia_model.keras"
#IMG_PATH = r"C:\Users\User\Desktop\normal1.jfif" 
#IMG_PATH = r"C:\Users\User\Desktop\pneumonia2bacterial.jfif"  
#IMG_PATH = r"C:\Users\User\Desktop\pneumonia3viral.jfif"  
IMG_PATH = r"C:\Users\User\Desktop\pneumoniafungal.avif"   

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
