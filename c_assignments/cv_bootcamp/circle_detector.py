import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('vw_bus.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20, param1=50, param2=70, minRadius=2, maxRadius=60);   

circled_img = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(circled_img, (i[0], i[1]), i[2], (0, 255, 255), 2)
        cv2.line(circled_img, (i[0] + int(i[2]/np.sqrt(2)), i[1] + int(i[2]/np.sqrt(2))), (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 10), (0, 255, 255), 3)
        cv2.line(circled_img, (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 10), (i[0] + int(i[2]/np.sqrt(2)) + 35, i[1] + int(i[2]/np.sqrt(2)) + 10), (0, 255, 255), 2)
        cv2.putText(circled_img, f"{i[2]}px", (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(circled_img, f"Detected circles: {len(circles[0])}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    print(f"Detected circles: {len(circles[0])}")

plt.imshow(cv2.cvtColor(circled_img, cv2.COLOR_BGR2RGB))
plt.title('Hough Circle Detection')
plt.axis('off')
plt.show()