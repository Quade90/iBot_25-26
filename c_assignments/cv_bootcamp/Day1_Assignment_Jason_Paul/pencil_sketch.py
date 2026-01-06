import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def pencil_sketch(img_color, kernel_size=(21, 21)):
    """Converts a color image to a pencil sketch and displays both images side by side.

    Parameters:
        img_color (numpy.ndarray): Input color image in BGR format.
        kernel_size (tuple): Kernel size for Gaussian blur.

    Returns:
        Pencil sketch image in grayscale and colour format.
    """

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # Convert to HSV color space
    img_hsv_h, img_hsv_s, img_hsv_v = cv2.split(img_hsv)

    img_hsv_s = (img_hsv_s * 0.75).astype(np.uint8) # Reduce saturation for grayscale effect

    img_blur = cv2.GaussianBlur(img_hsv_v, kernel_size, sigmaX=0, sigmaY=0) # Apply Gaussian blur
    img_sketch_v = cv2.divide(img_blur, img_hsv_v + 1, scale=256.0) # Blend hsv and inverted blur to create sketch
    img_sketch_v = cv2.normalize(img_sketch_v, None, 0, 255, cv2.NORM_MINMAX) # Normalize to full 0-255 range

    img_sketch_gray = img_sketch_v.astype(np.uint8)


    img_sketch = cv2.merge((img_hsv_h, img_hsv_s, img_sketch_v)) # Merge channels to create grayscale sketch
    img_sketch = cv2.cvtColor(img_sketch, cv2.COLOR_HSV2BGR)

    return img_sketch, img_sketch_gray

def pencil_sketch_video(video, kernel_size=(21, 21)):
    """#Bonus 3: Converts each frame of a video to a pencil sketch, both colour and grayscale and saves it.

    Parameters:
        video (cv2.VideoCapture): Input video capture object.
        kernel_size (tuple): Kernel size for Gaussian blur.

    Returns:
        None
    """
    video_handle = cv2.VideoCapture(str(video))

    if not video_handle.isOpened():
        raise ValueError("Could not open video file") # Handle invalid video file

    width = int(video_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_handle.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30 # Default fps if unable to get from video

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create outputs directory if it doesn't exist
    output_dir = Path("Outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{path.stem}_sketch_video.mp4"
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)) # Colour sketch video writer

    output_path = output_dir / f"{path.stem}_sketch_gray_video.mp4"
    video_writer_gray = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height)) # Grayscale sketch video writer

    # Process each frame
    while True:
        ret, frame = video_handle.read()
        if not ret:
            break
        img_sketch, img_sketch_gray = pencil_sketch(frame, kernel_size=kernel_size)
        video_writer.write(img_sketch)
        img_sketch_gray = cv2.cvtColor(img_sketch_gray, cv2.COLOR_GRAY2BGR)
        video_writer_gray.write(img_sketch_gray)

    video_handle.release()
    video_writer.release()
    video_writer_gray.release()

    print("Pencil sketch videos saved.")

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
    plt.imshow(cv2.cvtColor(img_sketch, cv2.COLOR_BGR2RGB))     
    plt.title("Pencil Sketch")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def kernel_size_input():
    """#Bonus1: Accepts user input for kernel size and validates it.

    Parameters:
        None

    Returns:
        kernel_size (tuple): Validated kernel size for Gaussian blur.
    """
    
    try:
        kernel_side = int(input("Enter kernel side length: "))
    except ValueError:
        kernel_side = 21
        print("Invalid input. Using default kernel size (21, 21).") # Default kernel size if input is not integer
    if kernel_side < 0:
        kernel_size = (21, 21) # Default kernel size if negative number is provided
        print("Negative number provided. Using default kernel size (21, 21).")
    elif kernel_side % 2 == 0:
        kernel_size = (21, 21) # Default kernel size if even number is provided
        print("Even number provided. Using default kernel size (21, 21).")
    else:
        kernel_size = (kernel_side, kernel_side)
    return kernel_size

def get_file_type(path):
    """Determines if the file is an image or video based on its extension.

    Parameters:
        path (Path): Path object of the file.

    Returns:
        str: "image", "video", or "unknown"
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    ext = path.suffix.lower()

    if ext in img_exts:
        return "image"
    elif ext in vid_exts:
        return "video"
    else:
        return "unknown"

input_dir = Path("Tests")

path_str = input("Enter file name: ")
path = input_dir / path_str

if path.exists():
    try:
        file_type = get_file_type(path)
        print("File exists: ", path)
        
        if file_type == "image":
            img_color = cv2.imread(path)
            if img_color is None:
                raise ValueError("OpenCV failed to load image")
            else:
                kernel_size = kernel_size_input()
                print("Image processing...")
                img_sketch, img_sketch_gray = pencil_sketch(img_color, kernel_size=kernel_size)
                tandem_display(img_color, img_sketch_gray)
                tandem_display(img_color, img_sketch)

                # Create outputs directory if it doesn't exist
                output_dir = Path("Outputs")
                output_dir.mkdir(exist_ok=True)

                # Save the colour sketch image
                output_path = output_dir / f"{path.stem}_sketch{path.suffix}"
                cv2.imwrite(str(output_path), img_sketch)

                # Save the grayscale sketch image
                output_path = output_dir / f"{path.stem}_sketch_gray{path.suffix}"
                cv2.imwrite(str(output_path), img_sketch_gray)

        elif file_type == "video":
            kernel_size = kernel_size_input()
            print("Video processing...")
            pencil_sketch_video(path, kernel_size=kernel_size)
        else:
            print("Unsupported file format.")
            exit()
    except cv2.error: # Handle OpenCV errors
        print("Processing error")
    
    except ValueError: # Handle invalid format
        print("Invalid format")

else:
    print("No such file exists.")

