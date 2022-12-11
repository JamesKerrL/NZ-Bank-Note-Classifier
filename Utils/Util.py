import cv2


def scale_down(img, percentage):
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)