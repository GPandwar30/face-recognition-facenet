import cv2
from .detection import recognize_faces

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()
            cv2.destroyAllWindows()

    def get_frame(self):
        while True:
            success, frame = self.video.read()
            if not success:
                break

            # Apply face recognition
            frame = recognize_faces(frame)

            # Encode the frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Yield frame in HTTP response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
