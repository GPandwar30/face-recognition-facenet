# Step 1: views.py in face_recog_app

from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponse
from django.conf import settings
import os
import cv2
from .camera import VideoCamera
from .detection import train_model
import threading
import time
from django.views.decorators.csrf import csrf_exempt

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(VideoCamera().get_frame(), content_type='multipart/x-mixed-replace; boundary=frame')

def enroll(request):
    if request.method == 'POST':
        username = request.POST['username']
        return redirect(f'/capture/?username={username}')
    return render(request, 'enroll.html')

def capture(request):
    username = request.GET.get('username')
    if not username:
        return HttpResponse("Username not provided", status=400)

    face_dir = os.path.join(settings.BASE_DIR, 'faces', username)
    os.makedirs(face_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return HttpResponse("Cannot open webcam", status=500)

    face_cascade_path = os.path.join(settings.BASE_DIR,'face_recog_app', 'model', 'haarcascade_frontalface_default.xml')
    if not os.path.exists(face_cascade_path):
        return HttpResponse("Face cascade file not found", status=500)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    count = 0
    while count < 150:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (160, 160))
            img_path = os.path.join(face_dir, f'image_{count:03d}.jpg')
            cv2.imwrite(img_path, resized)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow('Capturing...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    train_model()
    return HttpResponse(f"Captured {count} images for {username} and model trained.")





capture_counts = {}

def capture_page(request):
    return render(request, 'capture.html')

@csrf_exempt
def start_capture(request):
    username = request.POST.get('username')
    if not username:
        return HttpResponse("Username not provided", status=400)

    capture_counts[username] = 0
    thread = threading.Thread(target=capture_faces_thread, args=(username,))
    thread.start()

    return HttpResponse("Started")


def capture_faces_thread(username):
    face_dir = os.path.join(settings.BASE_DIR, 'faces', username)
    os.makedirs(face_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    while count < 150:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (160, 160))
            img_path = os.path.join(face_dir, f'image_{count:03d}.jpg')
            cv2.imwrite(img_path, resized)
            count += 1
            capture_counts[username] = count
            if count >= 150:
                break

    cap.release()
    cv2.destroyAllWindows()
    
    # ✅ Retrain model with all data after capture
    print("🔄 Retraining model with all users...")
    train_model()


def get_progress(request):
    username = request.GET.get('username')
    count = capture_counts.get(username, 0)
    return HttpResponse(str(count))