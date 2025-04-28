import cv2
import matplotlib.pyplot as plt

# 1) Initialize HOG descriptor with the default people detector SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2) Load your image (change path as needed)
img = cv2.imread('Test3.jpg')
if img is None:
    raise FileNotFoundError("Could not load image; check your path.")

# 3) Detect pedestrians
#    winStride: step size in x/y for the sliding window
#    padding: adds border pixels around each window (improves detection at edges)
#    scale: image pyramid scale factor
(rects, weights) = hog.detectMultiScale(
    img,
    winStride=(8, 8),
    padding=(8, 8),
    scale=1.05
)

# 4) Filter out weak detections by weight (confidence)
min_confidence = 0.6
strong = [
    (x, y, w, h) for (x, y, w, h), conf in zip(rects, weights)
    if conf > min_confidence
]

# 5) Draw the detections
output = img.copy()
for (x, y, w, h) in strong:
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 6) Display with matplotlib (convert BGR→RGB)
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(output_rgb)
plt.title(f"Pedestrians detected: {len(strong)}")
plt.axis('off')
plt.show()
