import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer
from mtcnn import MTCNN
from .architecture import InceptionResNetV2
from django.conf import settings


# Constants
REQUIRED_SIZE = (160, 160)
NORMALIZER = Normalizer('l2')
FACE_ENCODER = InceptionResNetV2()
FACE_ENCODER.load_weights(os.path.join('face_recog_app','model', 'facenet_keras_weights.h5'))
FACE_DETECTOR = MTCNN()


# Utility Functions
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def align_face_affine(face_img, left_eye, right_eye):
    left_eye_center = np.array(left_eye)
    right_eye_center = np.array(right_eye)

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        int((left_eye_center[0] + right_eye_center[0]) / 2),
        int((left_eye_center[1] + right_eye_center[1]) / 2)
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_face = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]),
                                  flags=cv2.INTER_CUBIC)

    return aligned_face

def recognize_faces(frame):
    from scipy.spatial.distance import cosine
    with open(os.path.join('face_recog_app','model', 'encodings.pkl'), 'rb') as f:
        encoding_dict = pickle.load(f)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = FACE_DETECTOR.detect_faces(img_rgb)

    for res in results:
        if res['confidence'] < 0.99:
            continue

        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_rgb[y1:y2, x1:x2]

        aligned = align_face_affine(face, res['keypoints']['left_eye'], res['keypoints']['right_eye'])
        aligned = normalize(aligned)
        aligned = cv2.resize(aligned, REQUIRED_SIZE)
        face_d = np.expand_dims(aligned, axis=0)

        encode = FACE_ENCODER.predict(face_d)[0]
        encode = NORMALIZER.transform(encode.reshape(1, -1))[0]

        name = "Unknown"
        min_dist = float("inf")

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < 0.5 and dist < min_dist:
                name = db_name
                min_dist = dist

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {min_dist:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

def train_model():
    face_data = os.path.join('faces')
    encoding_dict = {}

    for face_name in os.listdir(face_data):
        person_dir = os.path.join(face_data, face_name)
        encodes = []

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img_BGR = cv2.imread(image_path)

            if img_BGR is None:
                print(f"🚫 Unable to load image: {image_path}")
                continue

            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            detections = FACE_DETECTOR.detect_faces(img_RGB)

            if len(detections) == 0:
                print(f"🚫 No faces detected in: {image_path}")
                continue

            for det in detections:
                try:
                    x1, y1, width, height = det['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    face = img_RGB[y1:y2, x1:x2]

                    aligned = align_face_affine(face, det['keypoints']['left_eye'], det['keypoints']['right_eye'])
                    aligned = normalize(aligned)
                    aligned = cv2.resize(aligned, REQUIRED_SIZE)
                    face_d = np.expand_dims(aligned, axis=0)

                    encode = FACE_ENCODER.predict(face_d)[0]
                    encodes.append(encode)

                except Exception as e:
                    print(f"❌ Error processing face in {image_path}: {e}")

        if encodes:
            encode_sum = np.sum(encodes, axis=0)
            encode_norm = NORMALIZER.transform(np.expand_dims(encode_sum, axis=0))[0]
            encoding_dict[face_name] = encode_norm

    os.makedirs('model', exist_ok=True)
    with open(os.path.join('face_recog_app','model', 'encodings.pkl'), 'wb') as f:
        pickle.dump(encoding_dict, f)

    print("✅ Model training complete.")



def train_model():
    face_data = os.path.join('faces')
    encoding_dict = {}

    if not os.path.exists(face_data):
        print("⚠️ Faces directory not found.")
        return

    user_dirs = os.listdir(face_data)
    print("🔍 Found users:", user_dirs)

    for face_name in user_dirs:
        person_dir = os.path.join(face_data, face_name)
        if not os.path.isdir(person_dir):
            continue

        encodes = []
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img_BGR = cv2.imread(image_path)

            if img_BGR is None:
                print(f"🚫 Skipping unreadable image: {image_path}")
                continue

            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
            detections = FACE_DETECTOR.detect_faces(img_RGB)

            if len(detections) == 0:
                print(f"🚫 No face detected in: {image_path}")
                continue

            det = detections[0]  # Use the first face
            try:
                x1, y1, width, height = det['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2, x1:x2]

                aligned = align_face_affine(face, det['keypoints']['left_eye'], det['keypoints']['right_eye'])
                aligned = normalize(aligned)
                aligned = cv2.resize(aligned, REQUIRED_SIZE)
                face_d = np.expand_dims(aligned, axis=0)

                encode = FACE_ENCODER.predict(face_d)[0]
                encodes.append(encode)

            except Exception as e:
                print(f"❌ Error encoding {image_path}: {e}")
                continue

        if encodes:
            encode_sum = np.sum(encodes, axis=0)
            encode_norm = NORMALIZER.transform(np.expand_dims(encode_sum, axis=0))[0]
            encoding_dict[face_name] = encode_norm
            print(f"✅ Encoded {face_name}: {len(encodes)} samples.")
        else:
            print(f"⚠️ No encodings found for {face_name}")

    # Save encodings
    os.makedirs(os.path.join('face_recog_app', 'model'), exist_ok=True)
    enc_path = os.path.join('face_recog_app', 'model', 'encodings.pkl')
    with open(enc_path, 'wb') as f:
        pickle.dump(encoding_dict, f)

    print(f"✅ Model trained on {len(encoding_dict)} users. Encodings saved to {enc_path}.")
