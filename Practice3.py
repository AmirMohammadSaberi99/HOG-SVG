import cv2
import matplotlib.pyplot as plt

# 1) Load image
img_path = 'Test2.jpg'   # ⟵ change this to your file
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")

# Convert to gray for Haar
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2) HOG + SVM pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects_hog, weights) = hog.detectMultiScale(
    img,
    winStride=(8, 8),
    padding=(16, 16),
    scale=1.05
)
# filter by confidence
min_conf = 0.6
hog_dets = [r for r, w in zip(rects_hog, weights) if w > min_conf]

# 3) Haar cascade full‐body detector
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_fullbody.xml'
)
if haar_cascade.empty():
    raise IOError("Could not load haarcascade_fullbody.xml")
haar_dets = haar_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(60, 60)
)

# 4) Print counts
print(f"HOG+SVM detections   : {len(hog_dets)}")
print(f"Haar full‐body detections: {len(haar_dets)}")

# 5) Draw boxes on copies
img_hog  = img.copy()
img_haar = img.copy()

for (x, y, w, h) in hog_dets:
    cv2.rectangle(img_hog,  (x, y), (x+w, y+h), (0,255,0), 2)
for (x, y, w, h) in haar_dets:
    cv2.rectangle(img_haar, (x, y), (x+w, y+h), (255,0,0), 2)

# 6) Display side‐by‐side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Original', 'HOG + SVM (green)', 'Haar Full‐Body (blue)']
images = [img, img_hog, img_haar]

for ax, img_bgr, title in zip(axes, images, titles):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
