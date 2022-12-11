import cv2
import Utils.Constants as util

def internal_classify(detector, reference_image, image_to_classify):
    kp, descriptor = detector.detectAndCompute(reference_image, None)
    kp2, descriptor2 = detector.detectAndCompute(image_to_classify, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor, descriptor2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return len(good)


def classify(image_to_classify, threshold=100):
    number_of_matching_keypoints_list = []
    for name_and_path in util.REFERENCE_IMAGE_DESCRIPTIONS:
        reference_image = cv2.imread(name_and_path['path'])
        detector = cv2.xfeatures2d.SIFT_create()
        number_of_matching_keypoints = \
            internal_classify(detector, reference_image, image_to_classify)

        number_of_matching_keypoints_list.append(
            {'name': name_and_path['name'],
             'number_of_matches': number_of_matching_keypoints})

    maximum = max(number_of_matching_keypoints_list, key=lambda x:x['number_of_matches'])
    if maximum['number_of_matches'] > threshold:
        return maximum['name']
    return "No Match"
