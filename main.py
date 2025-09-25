import cv2
from preprocess import crop_board_auto, resize_to

# Load images
master = cv2.imread('master.jpg')
test = cv2.imread('test.jpg')

# Crop to the board
master_crop = crop_board_auto(master)
test_crop = crop_board_auto(test)

# Resize Test to match Master
test_aligned = resize_to(test_crop, master_crop.shape)

# Display results
cv2.imshow("Master Board", master_crop)
cv2.imshow("Test Aligned Board", test_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
