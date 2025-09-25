import cv2
import sys
import os
from preprocess import crop_board_auto, resize_to
from align import align_images
import numpy as np

# ==== Command-line args ====
if len(sys.argv) != 3:
    print("Usage: python main.py <master_image> <test_image>")
    sys.exit(1)

master_file = sys.argv[1]
test_file = sys.argv[2]

master = cv2.imread(master_file)
test = cv2.imread(test_file)
if master is None or test is None:
    print("Error: Could not load images.")
    sys.exit(1)

# ==== Step 1: Crop and flatten boards ====
master_crop, _ = crop_board_auto(master, debug=False, display_size=(800,800))
test_crop, _ = crop_board_auto(test, debug=False, display_size=(800,800))

# ==== Step 2: Automatic orientation correction on test BEFORE alignment ====
gray_master = cv2.cvtColor(master_crop, cv2.COLOR_BGR2GRAY)
candidates = {
    "original": test_crop,
    "h_flip": cv2.flip(test_crop, 1),
    "v_flip": cv2.flip(test_crop, 0),
    "rot_180": cv2.rotate(test_crop, cv2.ROTATE_180)
}

best_score = -1
best_img = test_crop
for img in candidates.values():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = np.corrcoef(gray_master.flatten(), gray.flatten())[0,1]
    if score > best_score:
        best_score = score
        best_img = img

test_crop = best_img

# ==== Step 3: Resize and align after flip correction ====
test_resized = resize_to(test_crop, master_crop.shape)
aligned_test = align_images(master_crop, test_resized)

# ==== Step 4: Color-based difference detection (Lab space) ====
master_lab = cv2.cvtColor(master_crop, cv2.COLOR_BGR2LAB)
aligned_lab = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2LAB)

diff_l = cv2.absdiff(master_lab[:,:,0], aligned_lab[:,:,0])
diff_a = cv2.absdiff(master_lab[:,:,1], aligned_lab[:,:,1])
diff_b = cv2.absdiff(master_lab[:,:,2], aligned_lab[:,:,2])

diff_color = cv2.max(cv2.max(diff_l, diff_a), diff_b)

# Dynamic threshold
mean_val = np.mean(diff_color)
std_val = np.std(diff_color)
k = 1.5
thresh_val = min(255, mean_val + k*std_val)
_, thresh = cv2.threshold(diff_color, thresh_val, 255, cv2.THRESH_BINARY)

# ==== Step 5: Highlight differences on TEST board ====
test_diff = aligned_test.copy()
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(test_diff, (x, y), (x + w, y + h), (0, 0, 255), 2)

# ==== Step 6: Display windows ====
cv2.imshow("Master Board", master_crop)
cv2.imshow("Test Board", test_crop)
cv2.imshow("Test Aligned + Differences", test_diff)
cv2.imshow("Difference Mask", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== Step 7: Save outputs ====
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/test_with_differences.jpg", test_diff)
cv2.imwrite("output/difference_mask.jpg", thresh)
