import cv2
import Classifier
VALIDATION_DIR = "ValidationDatasetImages"


def test_can_classify_5_dollar():
    image_to_classify = cv2.imread(VALIDATION_DIR + "/5 Dollar test image.jpg")
    assert "5 Dollar Note" == Classifier.classify(image_to_classify, threshold=1000)