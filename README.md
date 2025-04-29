
# 🌸 Flower Recognition using Deep Learning

This project implements a flower image classification system using MobileNetV2 and a stylish Tkinter-based GUI. It identifies five types of flowers: daisy, dandelion, rose, sunflower, and tulip.


## 📌 Project Description
This project focuses on building a deep learning-based flower classification system using transfer learning with MobileNetV2. The system is capable of identifying five distinct types of flowers—daisy, dandelion, rose, sunflower, and tulip—based on images. The model is trained on a labeled dataset of flower images, preprocessed and resized to a fixed dimension for optimal performance. Along with the training pipeline, a user-friendly Tkinter-based GUI is implemented to allow real-time flower recognition through image upload and classification. This project aims to demonstrate the effectiveness of convolutional neural networks (CNNs) in visual pattern recognition tasks and provide a seamless end-user experience.



## 📦 Requirements
- Python 3.7+
- TensorFlow
- OpenCV
- Pillow
- scikit-learn
- tqdm
- NumPy
- tkinter

Install them via pip:

```bash
pip install tensorflow opencv-python pillow scikit-learn tqdm
```


### 📁 Dataset Instructions
This project requires a flower image dataset for training and prediction. Due to GitHub's storage limitations, the dataset is not included in this repository.
**Please follow these steps to include the dataset locally:**
1. **Download the Dataset:**  
   Download the flower image dataset from the following link:  
   [Download Dataset from Google Drive](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)  

2. **Create a Folder:**  
   Inside the project root directory, create a folder named `flowers`.
3. **Add Category Folders:**  
   Inside the `flowers` folder, create the following subfolders:  
   ```
   flowers/
   ├── daisy/
   ├── dandelion/
   ├── rose/
   ├── sunflower/
   └── tulip/
   ```
4. **Place Images:**  
   Move the corresponding flower images into their respective subfolders.
Once this setup is complete, you can run the project and begin training or testing the flower recognition system.




### 🚀 Usage
1. **Clone the Repository**  
   Open a terminal and run the following commands:
   ```bash
   git clone https://github.com/yourusername/flower-recognition-project.git
   cd flower-recognition-project
   ```
2. **Install Required Packages**  
   Make sure you have Python installed. Then install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Dataset**  
   Place your flower dataset in the folder structure as follows:
   flower_project/
   └── flowers/
       ├── daisy/
       ├── dandelion/
       ├── rose/
       ├── sunflower/
       └── tulip/
   Each subfolder should contain relevant flower images.
4. **Train the Model (Optional)**  
   Run the main script:
   ```bash
   python flower_classifier.py
   ```
   If a saved model already exists, you will be asked whether to delete and retrain it.
5. **Launch the GUI**  
   After model loading or training is complete, a GUI window will open automatically. You can also start it manually:
   ```bash
   python flower_classifier.py
   ```
6. **Make Predictions**  
   Use the GUI to upload any flower image. The application will process the image and display the predicted flower type in real-time.



## 🧠 Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Top Layers:
  - GlobalAveragePooling2D
  - Dense(256, ReLU)
  - Dense(5, Softmax)
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Training:
  - 10 epochs (frozen base)
  - 10 epochs (fine-tuning with base trainable)



## 🖼️ Screenshot
![GUI Screenshot](<Screenshot 2025-04-29 223225.png>)



## 👨‍💻 Author
Developed as part of an academic mini-project in AI & ML.
