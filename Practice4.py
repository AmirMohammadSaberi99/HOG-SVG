import cv2
import matplotlib.pyplot as plt

# 1) Initialize HOG descriptor with the default people detector SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2) Load your image (change path as needed)
img = cv2.imread('Test3.jpg')
if img is None:
    raise FileNotFoundError("Could not load image; check your path.")
output = img.copy()

# 3) Detect pedestrians
#    detectMultiScale returns rectangles and corresponding SVM weights
(rects, weights) = hog.detectMultiScale(
    img,
    winStride=(8, 8),
    padding=(8, 8),
    scale=1.05
)

# 4) Draw bounding boxes + confidence
for (rect, weight) in zip(rects, weights):
    x, y, w, h = rect
    # confidence is the raw SVM score → map to [0,1] by a heuristic (e.g. sigmoid), or just display raw
    conf = float(weight)
    label = f"{conf:.2f}"
    # Draw box
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Draw label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(output, (x, y - text_h - 4), (x + text_w, y), (0, 255, 0), -1)
    # Put text
    cv2.putText(output, label, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# 5) Display via Matplotlib
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(output_rgb)
plt.title(f"Detected {len(rects)} pedestrians")
plt.axis('off')
plt.show()
