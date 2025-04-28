import cv2

def main():
    # 1) Replace 0 with the path to your video file
    VIDEO_PATH = 'Test.mp4'
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Couldn't open video {VIDEO_PATH}")
        return

    # 2) Initialize HOG descriptor + pre-trained people SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("Starting pedestrian detection on video. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            # end of file
            break

        # (Optional) Resize for speed
        # frame = cv2.resize(frame, (640, 360))

        # 3) Detect pedestrians
        rects, weights = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05
        )

        # 4) Filter by confidence
        min_confidence = 0.6
        strong = [(x, y, w, h) for (x, y, w, h), wgt in zip(rects, weights) if wgt > min_confidence]

        # 5) Draw boxes and count
        for (x, y, w, h) in strong:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"Pedestrians: {len(strong)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)

        # 6) Show
        cv2.imshow('Pedestrian Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
