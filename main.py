import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((150, 150))  # Adjust size as needed for your model
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array, img
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            return None, None
    return None, None

def predict(model, img_array):
    predictions = model.predict(img_array).flatten()
    return predictions

def on_predict():
    global model, label_result, image_label

    img_array, img = load_image()

    if img_array is not None:
        predictions = predict(model, img_array)
        result_str = f"Prediction: {'Pneumonia' if predictions[0] > 0.5 else 'Normal'}"
        label_result.config(text=result_str)

        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

def load_pneumonia_model():
    try:
        model = load_model('pneumonia_model.h5')
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return None

root = tk.Tk()
root.title("Pneumonia Detection")

model = load_pneumonia_model()

load_button = tk.Button(root, text="Load Image and Predict", command=on_predict)
load_button.pack()

image_label = tk.Label(root)
image_label.pack()

label_result = tk.Label(root, text="")
label_result.pack()

root.mainloop()
