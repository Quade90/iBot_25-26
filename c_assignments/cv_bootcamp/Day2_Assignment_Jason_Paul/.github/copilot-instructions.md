# Copilot Instructions for Circle Detector Project

## Project Overview
This is a single-file Python script (`circle_detector.py`) for detecting circles in images using OpenCV's Hough Circle Transform. It preprocesses images, detects circles, annotates them with adaptive text colors, and displays results with matplotlib.

## Key Patterns and Conventions
- **Image Handling**: Use OpenCV (cv2) for loading and processing images in BGR format. Convert to RGB for matplotlib display.
- **Preprocessing**: Always apply median blur (kernel size 5) after grayscale conversion to reduce noise before circle detection.
- **Hough Parameters**: Use tuned parameters like dp=1.2, minDist=50, param1=130, param2=80, minRadius=20, maxRadius=70. Adjust based on image characteristics.
- **Adaptive Text**: Implement `adaptive_text_color` function to choose dark green on bright areas, light green on dark areas, sampling a 20x20 box around text position.
- **Annotation Style**: Draw circles with green outline (thickness 2), red center dot, and a small radius indicator line. Label with format "(count)radius px" using FONT_HERSHEY_SIMPLEX size 0.35.
- **Statistics**: Calculate and print average, max, min radii for detected circles.
- **Error Handling**: Check file existence with pathlib.Path.exists(), handle OpenCV loading failures with try-except.

## Development Workflow
- **Running**: Execute `python circle_detector.py`, enter absolute image path when prompted. Script displays matplotlib figures and prints statistics.
- **Testing**: Manually test with different images. No automated tests exist; verify circle detection accuracy visually.
- **Dependencies**: Install via pip: numpy, opencv-python, matplotlib, pillow. Use virtual environment if needed.
- **Modifications**: When adding features, follow the modular function structure. For example, to add shape detection, create a new function similar to `detect_circles` and integrate into the main flow.

## Examples
- To change blur kernel: Modify `cv2.medianBlur(gray, 5)` to different odd size.
- To add ellipse detection: Use `cv2.fitEllipse` on contours, similar to circle drawing logic.
- To save output: Add `cv2.imwrite` after annotation, as commented in `detect_circles`.

This covers the core patterns for productive contributions.