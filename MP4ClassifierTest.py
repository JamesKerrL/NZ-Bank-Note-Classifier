import cv2
import Utils.Util as util
import Classifier


def draw_text(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 20)
    fontScale = 1
    fontColor = (0, 0, 0)
    thickness = 3
    lineType = 2

    cv2.putText(image, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

video_capture = cv2.VideoCapture('ValidationDataSetMP4/nz 20 dollar.mp4') # We turn the webcam on.

while True:
    _, frame = video_capture.read()
    text = Classifier.classify(frame)

    draw_text(frame, text)
    scaled_image = util.scale_down(frame, 50)
    cv2.imshow('Classified and scaled image', scaled_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()