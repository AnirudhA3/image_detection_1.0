import cv2
import numpy as np

def crop_board_auto(img, debug=False, display_size=(800,800)):
    """
    Crop the largest object (board), rotate to horizontal,
    and scale to fully fit the display window end-to-end.
    Uses minAreaRect for angle correction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, display_size), None

    image_area = img.shape[0] * img.shape[1]
    contours = [c for c in contours if cv2.contourArea(c) > 0.01 * image_area]
    if not contours:
        return cv2.resize(img, display_size), None

    largest_contour = max(contours, key=cv2.contourArea)

    # Rotated bounding rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.float32)

    width, height = int(rect[1][0]), int(rect[1][1])
    src_pts = box
    dst_pts = np.array([[0, height-1],
                        [0,0],
                        [width-1,0],
                        [width-1,height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # Resize to display window
    target_w, target_h = display_size
    scale = min(target_w / warped.shape[1], target_h / warped.shape[0])
    new_w, new_h = int(warped.shape[1]*scale), int(warped.shape[0]*scale)
    resized = cv2.resize(warped, (new_w, new_h))

    # Center in output window
    output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w)//2
    y_offset = (target_h - new_h)//2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    debug_img = output.copy() if debug else None
    return output, debug_img

def resize_to(img, target_shape):
    h, w = target_shape[:2]
    return cv2.resize(img, (w, h))
