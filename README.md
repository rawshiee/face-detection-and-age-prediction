👤 Face Detection, Age & Gender Prediction (OpenCV)

A Python-based computer vision project that detects human faces in images/video and predicts age and gender using pre-trained deep learning models with OpenCV DNN.

This repository is maintained mainly as a backup of an older project, but the code is functional and reusable.

✨ Features

Real-time face detection

Age prediction from detected faces

Gender prediction from detected faces

Uses OpenCV’s DNN module

Works with images, webcam, or video streams

🧠 Tech Stack

Python

OpenCV (cv2)

Pre-trained Caffe / TensorFlow models

📁 Project Structure
.
├── main.py
├── import cv2.py
├── loll.py
├── face detection file.py
├── models/
│   ├── age_deploy.prototxt
│   ├── gender_deploy.prototxt
│   ├── opencv_face_detector.pbtxt
│   └── model files (.caffemodel / .pb)
├── setup.bat
└── .gitignore

🚀 How to Run
1️⃣ Clone the repository
git clone https://github.com/rawshiee/YOUR_REPO.git
cd YOUR_REPO

2️⃣ Install dependencies
pip install opencv-python numpy

3️⃣ Run the project
python main.py


Ensure your webcam is connected for real-time detection.

👥 Collaborators

Rawshiee (@rawshiee
)

FriX (@frixisnotpeaceful
)

🗂️ Purpose of This Repository

Backup of an old computer vision project

Reference for OpenCV DNN-based face analysis

Learning resource for face detection pipelines

⚠️ Notes

Model files are pre-trained and large in size

Prediction accuracy depends on lighting and camera quality

Code structure may be cleaned in future revisions

📌 Future Improvements

Clean file naming and structure

Add requirements.txt

Improve visualization and UI

Train custom models

📜 License

This project is intended for educational and learning purposes.