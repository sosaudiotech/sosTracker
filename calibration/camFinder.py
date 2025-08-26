import cv2

for i in range(10):  # adjust range if needed
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print(f"Index {i}: {'OPEN' if ok else 'closed'}")
    if ok:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"probe_index_{i}.jpg", frame)
        cap.release()
