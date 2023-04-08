import cv2

face_cap = cv2.CascadeClassifier(
    "C:/Users/AL REHMAN LAPTOP/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default")
video_cap = cv2.VideoCapture(0)

while True:

    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    face = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, h, w) in face:
        cv2.rectangle(video_data, (x, y), (x + h, y + w), (0, 255, 0), 2)
    cv2.imshow("Live video", video_data)

video_cap.release()
