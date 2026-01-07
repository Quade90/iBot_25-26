# Image and Video to Pencil Sketch Converter

## Overview
This project converts **images and videos** into pencil-sketch styled outputs using OpenCV.  
It supports both **colour and grayscale sketches**, user-controlled blur strength, and an **interactive GUI** for real-time image preview and saving.

---

## How to Run

### 1. Start the program
Run the Python script normally.

### 2. Choose processing mode
When prompted:
- Type **`image`** to open the interactive GUI  
- Type **`video`** to process a video file

> **Note:** Video files must be placed inside the `Tests` folder.

---

## Image Mode (GUI)

- Click **Load Image** to select an image file
- Use the **Blur Kernel slider** to adjust sketch softness
- Toggle between **Colour** and **Grayscale** modes
- Preview updates live as the slider is adjusted
- Save results using:
  - **Save As** (custom location and filename)
  - **Quick Save** (automatically saved to `Outputs` folder)

Saved filenames follow the format:
- <original_name>_colour_sketch.<ext>
- <original_name>_grayscale_sketch.<ext>

---

## Video Mode (CLI)

- Enter the video filename when prompted
- Kernel size is requested via input
- Generates:
  - Colour pencil sketch video
  - Grayscale pencil sketch video
- Output videos are saved in the `Outputs` folder

---

## Features

- Image to pencil sketch conversion
- Video to pencil sketch conversion
- Colour and grayscale output support
- User-controlled blur kernel size (odd values enforced)
- Interactive GUI with live preview
- Automatic output folder creation
- Robust file and error handling
- Preserves original file extensions

---

## Dependencies

- Python 3.x  
- opencv-python  
- numpy  
- pillow  
- tkinter (included with standard Python installations)

---

## Observations

- Edges are detected clearly using intensity-based division techniques
- Shaded regions without sharp edges may appear smoother
- Larger kernel sizes produce softer sketches
- Smaller kernel sizes preserve fine details but may increase noise

---

## Challenges Faced

- Selecting an appropriate kernel size range
- Preventing image resizing feedback loops in the GUI
- Handling both CLI and GUI workflows cleanly
- Managing consistent file naming and output paths
- Ensuring stability across image and video processing
