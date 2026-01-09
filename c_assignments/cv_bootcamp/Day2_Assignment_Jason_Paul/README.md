# Image and Video Circle Detector

## Overview
This project detects circles in **images and videos** using the Hough Circle Transform with OpenCV.
It features an **interactive GUI** that allows users to tune detection parameters in real time, with the option to auto-tune for optimal results. Once circles are detected, a set of statistics related to the image is displayed. After suitable parameters are chosen, users can save the processed image along with its statistics file, or export the processed video.

--

## How to Run

### 1. Start the program
Run the python script normally.

### 2. Choose action
When prompted:
 - Click **`Load Image`** if image is desired
 - Click **`Load Video`** if video is desired

--

## Image 
- Upon clicking **`Load Image`** a dialog box will open and user must choose an image
- Once chosen the image will be auto-tuned and opened
- The user can then tweak the slider values if they choose to do so
- At any point the user can click **`Auto Tune`** to auto-tune parameters
- Once desirable changes are made the user can save the image and stats file by using:
    -**`Save as...`** to choose the save location
    -**`Quick save`** to save to the **Outputs** folder

Saved filenames follow the format
<original_name>_circled.png
<original_name>_stats.txt

--

## Video
- Upon clicking **`Load Video`** a dialog box will open and user must choose a video
- Once chosen the video will be auto-tuned and saved into the **Outputs** folder where the user can play it

--

## Features
- Image & video loading
- Automatic Hough parameter tuning
- Live slider controls (param2, min/max radius)
- Size-based circle coloring
    - Blue = small
    - Yellow = medium
    - Red = large
- Real-time stats display
- Quick Save & Save As options
- Exports detailed stats text file
- Frame-by-frame video processing
- Output video generation
- Built-in legend panel
- GUI built with Tkinter

--

## Dependencies
- Python 3.x
- OpenCV **`cv2`**
- NumPy
- Tkinter
- Pillow **`PIL`**

--

## Observations
- Hough circle detector is very sensitive to param2
- min_radius and max_radius can have an effect on already detected circles, even if they're radii are still within range, new circles can also form similarily
- Auto tuning helps massively, to instantly get best circles, or make small changes to and then get best circles

--

## Challenges
- Finding best Hough circle detector parameters
- Keeping image and UI from getting cut-off by window size
- Getting a clean video without undesirable circle detection
- Developing a clean UI layout


