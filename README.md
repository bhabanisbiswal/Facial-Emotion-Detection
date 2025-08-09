---

# 🖥 Facial Emotion Detection

This project performs **real-time facial emotion recognition** using **deep learning** 🧠 and **OpenCV** 📷.
It detects and classifies emotions such as **happy**, **sad**, **angry**, **surprised**, and **neutral** 😐 from facial expressions.

---

## ✨ Features

* 🎥 **Real-time emotion detection** from webcam feed.
* 🖼 **Static image emotion detection**.
* 📊 Recognizes **multiple emotions** – Happy, Sad, Angry, Surprised, Neutral.
* 🔍 Facial landmark detection for better accuracy.
* 🖥 Works on any system with a camera.

---

## 🛠 Tech Stack

* 🐍 **Python**
* 🎥 **OpenCV** – Real-time face detection and processing.
* 🔢 **NumPy** – Array operations and preprocessing.
* 🤖 **Deep Learning Model** – Trained for emotion classification.
* 📁 **Haar Cascade Classifier** – Face detection from images/video.

---

## 📂 Project Structure

```
Facial-Emotion-Detection/
│── FaceDetect_by_camera.py     # Real-time facial emotion detection from webcam
│── FaceDetect_by_image.py      # Facial emotion detection from images
│── face_with_landmarks.py      # Detects face landmarks
│── haarcascade_frontalface_default.xml # Haar Cascade model for face detection
│── demo.jpg                    # Sample image
│── demo1.jpg                   # Sample image
│── main.py                     # Main script for detection
│── README.md                   # Project documentation
│── .gitignore                  # Ignored files
```

---

## ⚙ How It Works

1️⃣ **Face Detection**

* Uses **Haar Cascade Classifier** to detect faces in frames.

2️⃣ **Preprocessing**

* Extracts the facial region and preprocesses it for the deep learning model.

3️⃣ **Emotion Classification**

* Predicts the emotion using a trained model.

4️⃣ **Output Display**

* Draws bounding boxes and displays detected emotion labels in real time.

---

## 📥 Installation

1. 📂 Clone the repository:

```bash
git clone https://github.com/bhabanisbiswal/Facial-Emotion-Detection.git
```

2. 📁 Navigate into the folder:

```bash
cd Facial-Emotion-Detection
```

3. 📦 Install dependencies:

```bash
pip install opencv-python numpy
```

4. ▶ Run the desired script:

```bash
python FaceDetect_by_camera.py   # Real-time detection
python FaceDetect_by_image.py    # Image-based detection
```

---

## 🚀 Usage

* 🎥 Ensure your webcam is connected and working.
* 💡 Maintain good lighting for accurate detection.
* 🖼 Use clear face images for image-based detection.
* ❌ Press `Q` to exit the real-time window.

---

## 📸 Demo

![image alt](demo.jpg)
![image alt](demo1.jpg)

---

## 🔮 Future Improvements

* 🤖 Integrate deep learning CNN models for higher accuracy.
* 📊 Add more emotion categories.
* 🌐 Deploy as a web app with Flask or Streamlit.

---

## 👤 Author

**Bhabani S Biswal** – Python & AI/ML Developer
📧 Email: [bhabanibiswalb17@gmail.com](mailto:bhabanibiswalb17@gmail.com)
🔗 GitHub: [Bhabani S Biswal](https://github.com/bhabanisbiswal)

---
