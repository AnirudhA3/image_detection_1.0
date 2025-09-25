import cv2
import numpy as np

def align_images(master, test):
    """
    Align test image to master using ORB + homography.
    Returns aligned image.
    """
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(master, None)
    kp2, des2 = orb.detectAndCompute(test, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("Warning: Not enough matches. Returning unaligned test.")
        return test

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        print("Warning: Homography failed. Returning unaligned test.")
        return test

    h, w = master.shape[:2]
    aligned = cv2.warpPerspective(test, M, (w, h))
    return aligned
