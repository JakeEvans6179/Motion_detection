Packages:
cv2 -- pip install opencv-python

numpy -- pip install numpy

serial -- pip install pyserial


Motion detection based on background subtraction

Issues faced was many boxes were formed when running the opencv Gaussian Mixture-based Background/Foreground Segmentation Algorithm (MOG2) as a result of camera auto exposure, white balance changes and noise
Issue was solved by implementing a reset function whereby if more than 25 bounding boxes are detected then we reset the process. Noise is reduced by implementing a minimum size of bounding box for them to be considered points of movement.
To plot an object we take centre points of all the bounding boxes and add them together in a weighted manner to find a single average centre point.
The object must be within a Euclidean distance of 30 pixels from the last frame to be valid and also must have existed for a minimum of three frames before being plotted.

This program successfully tracks single objects however if more than one object is moving in the frame it calculates the midpoint between them,
Testing fix (main.py) by using the midpoint of the largest blob to see if this allows for more accurate tracking.

Camera calibration - due to camera distortion and camera properties need to calibrate using opencv

GitHub repo to calibrate camera and find camera matrix and distortion coefficients: https://github.com/niconielsen32/camera-calibration

Allows us to undistort image and find the true angle of any pixel within the frame and sending the angles to a servo to aim and "fire" a laser pointer at source of movement
Currently attempting to make it work assuming a fixed distance (2m), if we want to accurately aim at different distances need to determine the distance from camera using triangulation (2 cameras).

use the board in this repo for calibration - set size to [CHESSBOARD_SIZE = (8, 6)] 
set resolution to camera resolution and calibrate

