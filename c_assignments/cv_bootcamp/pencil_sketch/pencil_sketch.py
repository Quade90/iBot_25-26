import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

def pencil_sketch(img_color):
    """Converts a color image to a pencil sketch and displays both images side by side.

    Parameters:
        img_color (numpy.ndarray): Input color image in BGR format.

    Returns:
        None
    """

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    img_gray_inv = 255 - img_gray

    img_blur = cv2.GaussianBlur(img_gray_inv, (71, 71), sigmaX=0, sigmaY=0) # Apply Gaussian blur
    img_blur_inv = 255 - img_blur
    img_sketch = cv2.divide(img_gray, img_blur_inv, scale=256.0) # Blend grayscale and inverted blur to create sketch

    output_path = path.with_name(f"{path.stem}_sketch{path.suffix}")
    cv2.imwrite(str(output_path), img_sketch)

    tandem_display(img_color, img_sketch)

def tandem_display(img_color, img_sketch):
    """Displays the original color image and the pencil sketch side by side.
    Parameters:
        img_color (numpy.ndarray): Original color image in BGR format.
        img_sketch (numpy.ndarray): Pencil sketch image in grayscale format.
    Returns:
        None
    """


    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_sketch, cmap="gray")     
    plt.title("Pencil Sketch")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

path = Path(input("Enter file path: "))

if path.exists():
    print("File exists: ", path)
    try:
        img_color = cv2.imread(path)
        if img_color is None:
            raise ValueError("OpenCV failed to load image")
        else:
            pencil_sketch(img_color)

    except cv2.error: # Handle OpenCV errors
        print("Image processing error")
    
    except ValueError: # Handle invalid image format
        print("Invalid image format")

else:
    print("No such file exists.")

