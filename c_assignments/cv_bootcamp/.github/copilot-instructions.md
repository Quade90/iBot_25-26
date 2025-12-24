# AI Coding Guidelines for CV Bootcamp Assignments

## Project Overview
This is a computer vision bootcamp codebase with Python scripts implementing image processing techniques using OpenCV. Scripts are numbered sequentially (e.g., `1_pencil_sketch.py`, `circle_detector.py`) and process specific image files located in the project root.

## Core Libraries and Imports
- **OpenCV (cv2)**: Primary library for image processing operations
- **NumPy**: Array operations and mathematical computations
- **Matplotlib**: Visualization and plotting (used for circle detection results)
- **Pandas**: Imported but not actively used in current scripts

Always import in this order: `pandas`, `numpy`, `cv2`, `matplotlib.pyplot`.

## Image Processing Patterns
- Images are read using `cv2.imread()` with filenames like `'firebird.jpg'`, `'vw_bus.jpg'`
- Color conversion: Use `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` for grayscale
- Blurring: Prefer `cv2.GaussianBlur()` for pencil sketch, `cv2.medianBlur()` for circle detection
- Display: Use `cv2.imshow()` for intermediate results, `plt.show()` for final matplotlib plots

## Specific Techniques
### Pencil Sketch Effect ([1_pencil_sketch.py](1_pencil_sketch.py))
- Convert to grayscale → invert → Gaussian blur → invert → divide by blurred inverted image
- Scale factor of 256.0 in division for proper contrast
- Display both original and processed images side-by-side

### Circle Detection ([circle_detector.py](circle_detector.py))
- Use `cv2.HoughCircles()` with parameters: `dp=1.0`, `minDist=20`, `param1=50`, `param2=70`, `minRadius=2`, `maxRadius=60`
- Draw detected circles with yellow outline and red center dot
- Convert BGR to RGB before matplotlib display: `cv2.cvtColor(circled_img, cv2.COLOR_BGR2RGB)`

## File Organization
- Scripts are named with leading numbers for sequence
- Image assets stored in project root alongside scripts
- No modular structure - each script is self-contained

## Execution Assumptions
- Images exist in the same directory as scripts
- OpenCV windows require `cv2.waitKey(0)` and `cv2.destroyAllWindows()` for proper cleanup
- Matplotlib plots use `plt.axis('off')` to hide axes on image displays

## Development Workflow
- Run scripts directly: `python script_name.py`
- No build system or virtual environment specified
- Debug by adding intermediate `cv2.imshow()` calls for image processing steps