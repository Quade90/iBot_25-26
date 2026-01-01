# Project Title
Simple image-to-pencil-sketch converter

## How to run:
1. Run program.
2. When prompted enter file path.
3. Pencil sketch will be generated and saved alongside the original image.

## Dependencies
- Python 3.x
- opencv-python
- numpy
- matplotlib
- pillow

## Observations
- Using this method, edges are detected well but not shaded regions with non-exact edges.
- Larger kernel size results in softer, small kernel sizes result in less detail in some areas.

## Challenges
- Finding apt kernel size
- Handling exceptions
