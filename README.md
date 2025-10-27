# 👤 Face Recognition System (Django + OpenCV + Deep Learning)

A **web-based real-time face recognition system** built using **Django**, **OpenCV**, and **InceptionResNetV2**.  
It detects, recognizes, and tracks faces through a webcam feed, and stores face data in a Django-managed database for authentication or attendance systems.

---

## 🚀 Features

- 🎥 **Live webcam streaming** integrated into a Django web interface  
- 🧠 **Face recognition using InceptionResNetV2 embeddings**  
- 👁️ **Face detection using MTCNN**  
- 💾 **Stores trained encodings with pickle**  
- 🧩 **Multi-threaded video streaming** for smooth real-time processing  
- 🔐 **Django backend** for face data management and user control  
- 🧰 Option to **train/update models dynamically** from the web interface  

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| Framework | Django |
| Language | Python |
| Computer Vision | OpenCV |
| Deep Learning | InceptionResNetV2, MTCNN |
| Data Processing | NumPy, scikit-learn |
| Database | SQLite / MySQL |
| Frontend | HTML, CSS, Bootstrap |
| Threading | Python `threading` |
| Deployment | XAMPP / Localhost / Render *(optional)* |

---

## 📂 Project Structure

```
face_recognition_system/
│
├── face_app/
│   ├── templates/face_app/
│   │   ├── index.html
│   │   └── camera.html
│   ├── static/face_app/
│   │   └── (CSS, JS, images)
│   ├── camera.py             # Video stream handling
│   ├── detection.py          # Face detection and training logic
│   ├── views.py              # Main Django views
│   ├── models.py             # Face data models
│   ├── urls.py
│   └── ...
│
├── face_recognition_system/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── media/                    # Uploaded/processed images
├── db.sqlite3
├── manage.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-recognition-system.git
   cd face-recognition-system
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Run the server**
   ```bash
   python manage.py runserver
   ```
   Visit 👉 `http://127.0.0.1:8000/`

---

## 🧠 Model Architecture

The system combines **classical and deep learning components**:

1. **MTCNN (Face Detection)** – detects faces and keypoints in images or video frames.  
2. **InceptionResNetV2 (Embedding Extraction)** – extracts 128D face embeddings.  
3. **Normalizer (scikit-learn)** – standardizes embeddings before comparison.  
4. **Pickle-based Storage** – saves and loads known face embeddings.  
5. **Matching Algorithm** – compares current face encodings with known ones using cosine similarity.

---

## 💡 How It Works

1. The camera feed is accessed via the `VideoCamera` class using OpenCV.  
2. Detected faces are cropped and passed through **InceptionResNetV2** to extract embeddings.  
3. Embeddings are normalized and compared against known encodings stored in a pickle file.  
4. If a match is found → the person’s name is displayed.  
5. Results are rendered on the Django frontend in real-time.

---

## 📊 Example Output

```
[INFO] Starting camera stream...
[INFO] Detected: Gaurav Pandwar
[INFO] Unknown face detected
```

---

## 🧪 Future Enhancements

- 🔒 Add attendance tracking and CSV export  
- ☁️ Store face encodings in a database or cloud  
- 📱 Integrate mobile camera streaming  
- ⚡ Add GPU acceleration (TensorFlow / CUDA)

---

## 👨‍💻 Author

**Gaurav Pandwar**  
📧 [gauravpandwar@gmail.com](mailto:gp3084@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/gp30) | [GitHub](https://github.com/GPandwar30/face-recognition-facenet/)

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use and modify it.
