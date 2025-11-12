Packages:
cv2 -- pip install opencv-python

numpy -- pip install numpy

serial -- pip install serial




Motion detection based on background subtraction

Issues faced was many boxes were formed when running the opencv Gaussian Mixture-based Background/Foreground Segmentation Algorithm (MOG2) as a result of camera auto exposure, white balance changes and noise
Issue was solved by implementing a reset function whereby if more than 25 bounding boxes are detected then we reset the process. Noise is reduced by implementing a minimum size of bounding box for them to be considered points of movement.
To plot an object we take centre points of all the bounding boxes and add them together in a weighted manner to find a single average centre point.
The object must be within a eucladian distance of 30 pixels from the last frame to be valid and also must have existed for a minimum of three frames before being plotted.

This program successfully tracks single objects however if more than one object is moving in the frame.
