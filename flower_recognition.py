import os
import numpy as np
import tensorflow as tf
import threading
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2
from tqdm import tqdm

# ------------------ Settings ------------------
BASE_DIR = 'C:/Users/ADMIN/OneDrive/Desktop/flower_project/flowers/'
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = 128
MODEL_PATH = 'flower_model.keras'
# ------------------------------------------------

# Load dataset
def load_dataset():
    X, Z = [], []
    for category in CATEGORIES:
        folder = os.path.join(BASE_DIR, category)
        for img_name in tqdm(os.listdir(folder), desc=f"Loading {category}"):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            Z.append(category)
    le = LabelEncoder()
    Y = le.fit_transform(Z)
    Y = to_categorical(Y, num_classes=len(CATEGORIES))
    X = np.array(X) / 255.0
    return train_test_split(X, Y, test_size=0.2, random_state=42)

# Build model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Ask user whether to retrain
def check_retrain():
    if os.path.exists(MODEL_PATH):
        choice = input("\n[QUESTION] Existing model found. Do you want to delete and retrain? (y/n): ").strip().lower()
        if choice == 'y':
            print("[INFO] Deleting old model...")
            os.remove(MODEL_PATH)
            return True
        else:
            print("[INFO] Keeping existing model.")
            return False
    else:
        return True

# Train or load model
retrain = check_retrain()

if retrain:
    print("[INFO] Training new model for 20 epochs...")
    x_train, x_test, y_train, y_test = load_dataset()
    model = create_model()

    # Initial training (frozen base)
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # Fine-tuning (unfreeze base)
    model.layers[1].trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    model.save(MODEL_PATH)
    print(f"[INFO] Model saved as '{MODEL_PATH}'")
else:
    model = load_model(MODEL_PATH)

# Predict function
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return CATEGORIES[class_idx]

# Stylish GUI
def stylish_gui():
    def choose_image():
        file_path = filedialog.askopenfilename(title="Select a Flower Image",
                                                filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((250, 250))
            img = ImageTk.PhotoImage(img)
            panel.configure(image=img)
            panel.image = img

            # Loading spinner
            result_label.config(text="⏳ Predicting...", fg="#fdcb6e")
            window.update()

            threading.Thread(target=predict_and_display, args=(file_path,)).start()

    def predict_and_display(file_path):
        predicted = predict_flower(file_path)
        result_label.config(text=f"Prediction: {predicted}", fg="#00b894", font=("Arial", 16, "bold"))

    window = tk.Tk()
    window.title("🌸 Flower Classifier")
    window.geometry('400x500')
    window.config(bg="#dfe6e9")

    title = Label(window, text="🌻 Flower Recognition System", bg="#dfe6e9",
                  fg="#0984e3", font=("Arial", 20, "bold"))
    title.pack(pady=10)

    panel = Label(window, bg="#dfe6e9")
    panel.pack(pady=10)

    select_btn = Button(window, text="Select Image", command=choose_image,
                        bg="#74b9ff", fg="white", font=("Arial", 14, "bold"),
                        activebackground="#0984e3", cursor="hand2", relief="raised", bd=5)
    select_btn.pack(pady=10)

    result_label = Label(window, text="", bg="#dfe6e9", fg="#00b894", font=("Arial", 16, "bold"))
    result_label.pack(pady=20)

    window.mainloop()

# Start GUI
if __name__ == "__main__":
    stylish_gui()
