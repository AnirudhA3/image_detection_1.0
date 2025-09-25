import cv2
import numpy as np

def crop_board_auto(img):
    """
    Robustly crop the largest object (assumed board) in the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img  # fallback

    # Filter out small contours
    image_area = img.shape[0] * img.shape[1]
    contours = [c for c in contours if cv2.contourArea(c) > 0.05 * image_area]  # >5% of image area

    # Take largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate to polygon and crop bounding rectangle
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    cropped = img[y:y+h, x:x+w]

    return cropped

def resize_to(img, target_shape):
    """
    Resize image to match target shape.
    """
    h, w = target_shape[:2]
    return cv2.resize(img, (w, h))
