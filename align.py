import cv2
import numpy as np

def align_images(master, test):
    """
    Align test image to master using ORB + homography.
    Safely handles cases with fewer than 4 matches.
    """
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(master, None)
    kp2, des2 = orb.detectAndCompute(test, None)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Need at least 4 matches to compute homography
    if len(matches) < 4:
        print("Warning: Not enough matches to compute homography. Returning unaligned image.")
        return test

    # Extract matched points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

    # Compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Fallback if homography fails
    if M is None:
        print("Warning: Homography computation failed. Returning unaligned image.")
        return test

    h, w = master.shape[:2]
    aligned = cv2.warpPerspective(test, M, (w, h))
    return aligned
