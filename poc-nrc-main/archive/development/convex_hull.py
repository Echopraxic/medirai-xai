#%%
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)
import os
import matplotlib.pyplot as plt

# %%
def thresh_callback(thresh, src_gray):
    canny_output = cv.Canny(src_gray, thresh, thresh * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)

    # Find the outermost contour
    def bounding_area(c):
        x, y, w, h = cv.boundingRect(c)
        return w * h

    outermost_contour = max(contours, key=bounding_area)
    hull = cv.convexHull(outermost_contour)

    drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
    cv.drawContours(drawing, [hull], -1, (255, 255, 255), thickness=cv.FILLED)

    return drawing
    
#%%
image_folder = ".\data\convexhull"
output_folder = ".\data\convexhull_processed"
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    img = cv.imread(image_path)
    src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    thresh = 100
    result_img = thresh_callback(thresh, src_gray)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result_rgb = cv.cvtColor(result_img, cv.COLOR_BGR2RGB)

    output_path = os.path.join(output_folder, filename)
    cv.imwrite(output_path, result_img)    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f'Original: {filename}')
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Convexhull')
    plt.imshow(result_rgb)
    plt.axis('off')

    plt.show()
# %%
