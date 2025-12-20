import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

imgC = cv2.imread('firebird.jpg')
imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY) # Convert to grayscale
imgG_inv = 255 - imgG # Invert the grayscale image

imgB = cv2.GaussianBlur(imgG_inv, (21, 21), sigmaX=0, sigmaY=0) # Apply Gaussian Blur
imgB_inv = 255 - imgB # Invert the blurred image

imgS = cv2.divide(imgG, imgB_inv, scale=256.0) # Create the pencil sketch effect

cv2.imshow('Original Image', imgC)
cv2.imshow('Pencil Sketch', imgS)
cv2.waitKey(0)
cv2.destroyAllWindows()