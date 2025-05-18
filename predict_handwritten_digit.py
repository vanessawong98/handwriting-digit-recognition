import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model('handwritten_digit_model.keras')

print('Shape of model:: ')
print(model.input_shape)

def predict_image(image_path):
    try:
        img = Image.open(image_path).convert('L')

        img = img.resize((28, 28))

        img_array = np.array(img) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_digit = np. argmax(prediction)
        confidence = np.max(prediction) * 100

        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {predicted_digit}\nConfidence: {confidence:.2f}%")
        plt.axis('off')
        plt.savefig('predicted_img.png')

        print(f"Successfully predicted: {predicted_digit} (Confidence: {confidence:.2f}%)")

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"Error loading or processing the image: {e}")

if __name__ == '__main__':
    image_file = input("Enter the path to your PNG image: ")
    if os.path.exists(image_file):
        predict_image(image_file)
    else: 
        print("Invalid file path")
