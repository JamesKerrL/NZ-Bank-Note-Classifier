import cv2
import Utils.Util as util

def detect_rectangles(image, thresholdL=0, thresholdU=60):
    blurred = cv2.blur(image, (7, 7), 1)
    greyed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(blurred, thresholdL, thresholdU)

    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                cv2.drawContours(canny, cnt, -1, (255, 0, 255), 7)

    return canny

def empty():
    pass

cv2.namedWindow("Controls")
cv2.resizeWindow("Controls", 600, 200)
cv2.createTrackbar("ThresholdL","Controls", 100, 250, empty)
cv2.createTrackbar("ThresholdU","Controls", 100, 250, empty)

video_capture = cv2.VideoCapture('ValidationDataSetMP4/nz 20 dollar.mp4') # We turn the webcam on.

while True:
    _, frame = video_capture.read()

    thresholdL = cv2.getTrackbarPos("ThresholdL", "Controls")
    thresholdU = cv2.getTrackbarPos("ThresholdU", "Controls")
    img = detect_rectangles(frame)
    scaled_image = util.scale_down(img, 50)
    cv2.imshow('Classified and scaled image', scaled_image)
    if cv2.waitKey(90) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()