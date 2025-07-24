# 🔥 Fire Detection Using ResNet18 with Fine-Tuning & Streamlit Deployment

This project is a deep learning solution for detecting fire in images. It leverages the **ResNet18** architecture with **transfer learning and fine-tuning** to improve performance. The model is deployed using **Streamlit**, providing a simple web-based interface for users to upload an image and receive fire detection results.

---

## 📁 Project Structure

├── app.py # Streamlit web app

├── Fire_detection_model.h5 # Trained ResNet18 model (NOT INCLUDED – see below)

├── requirements.txt # Required dependencies

└── README.md # Project documentation

---

## 📌 Features

- ✅ Fine-tuned **ResNet18** pretrained on ImageNet  
- ✅ Applied **data augmentation**, **dropout**, and **batch normalization**  
- ✅ Deployed via **Streamlit** with an image upload interface  
- ✅ Fire detection results with simple prediction output  
- ✅ Lightweight and fast

---

## 🧠 Model Training Overview

- **Base model:** ResNet18  
- **Modifications:**
  - Removed top layers  
  - Added custom dense layers  
  - Included Dropout & BatchNormalization  
- **Optimization:**
  - Adam optimizer  
  - Data augmentation for generalization  
- **Loss:** binary crossentropy 

---

## 🔗Download The Model 
 (Not Included)


