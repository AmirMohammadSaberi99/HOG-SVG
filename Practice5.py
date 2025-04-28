import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    # convert to float if not already
    boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]

    # sort by score (ascending)
    idxs = np.argsort(scores)

    # greedily keep highest‐scoring boxes
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        # compute intersection with the rest
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        # areas
        area_last = (x2[last] - x1[last]) * (y2[last] - y1[last])
        area_others = (x2[idxs[:-1]] - x1[idxs[:-1]]) * (y2[idxs[:-1]] - y1[idxs[:-1]])

        # IoU
        iou = inter / (area_last + area_others - inter)

        # delete all indexes with IoU > threshold
        suppress = np.concatenate(([len(idxs)-1], np.where(iou > overlapThresh)[0]))
        idxs = np.delete(idxs, suppress)

    return boxes[pick].astype("int")


def pyramid(image, scale=1.25, minSize=(64, 128)):
    # Yield successive resized images
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize[1]):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize[0]):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == "__main__":
    # 1) Prepare HOG + get SVM weights & bias
    hog = cv2.HOGDescriptor()
    svm_detector = hog.getDefaultPeopleDetector()
    weights = np.array(svm_detector[:-1], dtype=np.float32)
    bias    = svm_detector[-1]

    # 2) Parameters
    windowSize = hog.winSize           # typically (64,128)
    stepSize   = (8, 8)                # sliding window stride
    pyramidScale = 1.25
    scoreThreshold = 0.0               # keep windows with score > 0.0
    nmsThreshold   = 0.3               # IoU for non-max suppression

    # 3) Load & preprocess
    orig = cv2.imread('Test2.jpg')  # ⟵ change this to your file
    if orig is None:
        raise FileNotFoundError("Image not found.")
    detections = []

    # 4) Loop over image pyramid
    for resized in pyramid(orig, scale=pyramidScale, minSize=windowSize):
        scaleFactor = orig.shape[1] / float(resized.shape[1])

        # 5) Loop over sliding windows
        for (x, y, window) in sliding_window(resized, stepSize, windowSize):
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue

            # 6) Compute HOG descriptor & score via dot + bias
            descriptor = hog.compute(window).flatten()
            score = float(np.dot(weights, descriptor) + bias)

            if score > scoreThreshold:
                # map window back to original image coordinates
                x1 = int(x * scaleFactor)
                y1 = int(y * scaleFactor)
                x2 = int((x + windowSize[0]) * scaleFactor)
                y2 = int((y + windowSize[1]) * scaleFactor)
                detections.append([x1, y1, x2, y2, score])

    # 7) Apply non-max suppression
    if len(detections) > 0:
        dets = np.array(detections)
        picks = non_max_suppression_fast(dets, nmsThreshold)
    else:
        picks = []

    # 8) Draw final boxes
    output = orig.copy()
    for (x1, y1, x2, y2, score) in picks:
        label = f"{score:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    # 9) Display
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(output_rgb)
    plt.title(f"Detections after NMS: {len(picks)}")
    plt.axis('off')
    plt.show()
