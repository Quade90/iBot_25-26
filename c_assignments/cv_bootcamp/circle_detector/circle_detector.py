import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def preprocess_image(img):
    """Convert image to grayscale and apply median blur to reduce noise.

    Parameters:
        img (numpy.ndarray): Input color image.

    Returns:
        numpy.ndarray: Preprocessed grayscaled and blurred image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blur = cv2.medianBlur(gray, 5)
    return blur

def adaptive_text_color(img, x, y, box=20, thresh=127):
    """Determine adaptive text color based on local brightness.

    Parameters:
        img (numpy.ndarray): Input color image.
        x (int): X-coordinate of the text position.
        y (int): Y-coordinate of the text position.
        box (int): Size of the box around the text position to sample brightness.
        thresh (int): Brightness threshold to decide text color.

    Returns:
        tuple: BGR color tuple for text."""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Create local box around (x, y)
    h, w = gray.shape
    x1 = max(0, x - box)
    y1 = max(0, y - box)
    x2 = min(w, x + box)
    y2 = min(h, y + box)

    mean_intensity = np.mean(gray[y1:y2, x1:x2]) # Calculate mean intensity in the local box

    if mean_intensity > thresh:
        DARK_GREEN  = (0, 100, 0) # Dark text on bright area
        return DARK_GREEN      
    else:
        LIGHT_GREEN = (144, 238, 144) # Light text on dark area
        return LIGHT_GREEN   

def detect_circles(img, blur):
    """Detect and label circles in the image using Hough Circle Transform.
    
    Parameters:
        img (numpy.ndarray): Original color image.
        blur (numpy.ndarray): Preprocessed grayscaled and blurred image.

    Returns:
        numpy.ndarray: Image with detected circles labeled.
        int: Number of detected circles.
        list: List of detected circles with their parameters."""
    
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Invalid input image")

    if blur is None or len(blur.shape) != 2:
        raise ValueError("Blurred image must be grayscale")
    
    # Hough Circle Transform parameters
    HOUGH_PARAMS = {
    "dp": 1.2,
    "minDist": 50,
    "param1": 139,
    "param2": 75,
    "minRadius": 20,
    "maxRadius": 120
    }
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    HOUGH_PARAMS["dp"],
    HOUGH_PARAMS["minDist"],
    param1=HOUGH_PARAMS["param1"],
    param2=HOUGH_PARAMS["param2"],
    minRadius=HOUGH_PARAMS["minRadius"],
    maxRadius=HOUGH_PARAMS["maxRadius"]
    )

    circled_img = img.copy()

    # Draw detected circles and labels
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for count, i in enumerate(circles[0, :]):
            tx = i[0] + int(i[2] / np.sqrt(2)) + 10
            ty = i[1] + int(i[2] / np.sqrt(2)) + 5
            color = adaptive_text_color(circled_img, tx, ty)
            
            cv2.circle(circled_img, (i[0], i[1]), i[2], color, 2)
            cv2.circle(circled_img, (i[0], i[1]), 2, (0, 0, 255), -1)
            cv2.line(circled_img, (i[0] + int(i[2]/np.sqrt(2)), i[1] + int(i[2]/np.sqrt(2))), (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 10), color, 3)
            cv2.line(circled_img, (i[0] + int(i[2]/np.sqrt(2)) + 10, i[1] + int(i[2]/np.sqrt(2)) + 10), (i[0] + int(i[2]/np.sqrt(2)) + 55, i[1] + int(i[2]/np.sqrt(2)) + 10), color, 2)
            cv2.putText(circled_img, f"({count + 1}){i[2]}px", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        
        color = adaptive_text_color(circled_img, 50, 50)
        cv2.putText(circled_img, f"Detected circles: {len(circles[0])}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)    
        
        # Save output image
        output_path = path.with_name(f"{path.stem}_circled{path.suffix}")
        cv2.imwrite(str(output_path), circled_img)

        # Show statistics
        print(f"Number of circles detected: {len(circles[0])}")
        radius_sum = 0
        max_radius = 0
        min_radius = 1e6
        for count, i in enumerate(circles[0, :]):
            print(f"{count + 1}.pos = ({i[0]},{i[1]}), r = {i[2]}px")
            radius_sum += i[2]
            max_radius = i[2] if i[2] > max_radius else max_radius
            min_radius = i[2] if i[2] < min_radius else min_radius
        print(f"Average radius: {radius_sum/len(circles[0]):.2f}px")
        print(f"Max radius: {max_radius}px")
        print(f"Min radius: {min_radius}px")

        #Export statistics to a text file
        stats_path = path.with_name(f"{path.stem}_stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"Number of circles detected: {len(circles[0])}\n")
            for count, i in enumerate(circles[0, :]):
                f.write(f"{count + 1}.pos = ({i[0]},{i[1]}), r = {i[2]}px\n")
            f.write(f"Average radius: {radius_sum/len(circles[0]):.2f}px\n")
            f.write(f"Max radius: {max_radius}px\n")
            f.write(f"Min radius: {min_radius}px\n")

        return circled_img, len(circles[0]), circles
    else:
        return img, 0, [[]]

def tandem_display(img, circled_img):
    """Displays the original image and the circled image side by side.

    Parameters:
        img (numpy.ndarray): Original image in BGR format.
        circled_img (numpy.ndarray): Image with detected circles.

    Returns:
        None
    """

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(circled_img, cv2.COLOR_BGR2RGB)) 
    plt.title("Circled Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

path = Path(input("Enter file path: "))

if path.exists():
    print("File exists: ", path)
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError("OpenCV failed to load image")
        else:
            circled_img, num_circles, circles = detect_circles(img, preprocess_image(img))

            if num_circles == 0:
                print("No circles detected.")
            
            tandem_display(img, circled_img)    

    except cv2.error: # Handle OpenCV errors
        print("Image processing error")
    
    except ValueError: # Handle invalid image format
        print("Invalid image format")

else:
    print("No such file exists.")



