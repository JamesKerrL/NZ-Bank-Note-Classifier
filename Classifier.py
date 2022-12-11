import cv2


def classify(reference_image, image_to_classify):
    detector = cv2.xfeatures2d.SIFT_create()

    kp, descriptor = detector.detectAndCompute(reference_image, None)
    kp2, descriptor2 = detector.detectAndCompute(image_to_classify, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor, descriptor2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return len(good) > 300