import cv2
name = 'video=Logitech HD Webcam C270'  # paste your exact string
for backend in [cv2.CAP_DSHOW, cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_ANY]:
    cap = cv2.VideoCapture(name, backend)
    print('backend', backend, 'opened?', cap.isOpened())
    cap.release()
