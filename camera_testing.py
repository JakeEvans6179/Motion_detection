import cv2 as cv
import numpy as np

"""
Used to find the ideal exposure and brightness threshold to apply to the camera for the sweeping laser test to get data to run linear regression
Run and shine laser at a flat surface at a known distance away, change values until only the laser is detected and use those values for the calibration_ml_script.py
"""

#camera matrix and distortion matrix
K = np.array([[5.502101709080033061e+02, 0, 3.229873787700622074e+02],
                    [0, 5.509288296445998867e+02, 2.321190112338632616e+02],
                    [0, 0, 1]], dtype=np.float32)
D = np.array([4.391641351825171374e-02, -2.803020736590210449e-01,-1.217892293841370006e-03, -1.528847043158600670e-03, 3.432285588703486434e-01], dtype=np.float32)


# Helper function for trackbars
def nothing(x):
    pass


# Initialize Camera
cap = cv.VideoCapture(0)

# Create a Window
window_name = "Camera Setup & Laser Check"
cv.namedWindow(window_name)

# --- CREATE TRACKBARS ---
# Exposure: OpenCV usually treats exposure as negative numbers on Windows (-1 to -13)
# We map the slider 0-20 to be -10 to +10 roughly, or strictly negative.
# Adjust 'val - 14' logic below depending on your specific camera driver.
cv.createTrackbar("Exposure Level", window_name, 4, 15, nothing)

# Threshold: How bright must the dot be to be counted?
cv.createTrackbar("Min Brightness", window_name, 200, 255, nothing)

print("Controls:")
print("1. Lower 'Exposure Level' until the room is BLACK.")
print("2. Point your laser at the wall.")
print("3. Ensure 'Max Val' is > 'Min Brightness'.")
print("Press 'q' to quit.")

while True:
    # 1. GET SLIDER VALUES
    exp_slider = cv.getTrackbarPos("Exposure Level", window_name)
    min_brightness = cv.getTrackbarPos("Min Brightness", window_name)

    # 2. SET EXPOSURE
    # Note: Different cameras handle this property differently.
    # Common values are -1 to -10. We map slider (0 to 15) to (-11 to 4)
    exposure_val = exp_slider - 11

    # Turning off Auto Exposure is critical first
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = Manual mode (for many drivers)
    cap.set(cv.CAP_PROP_EXPOSURE, exposure_val)

    # 3. READ FRAME
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    #undistort

    frame = cv.undistort(frame, K, D)

    # 5. FIND LASER (The Logic)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the brightest single pixel
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)

    # Create a visual mask to see what is passing the threshold
    _, thresh_mask = cv.threshold(gray, min_brightness, 255, cv.THRESH_BINARY)

    # 6. VISUALIZATION
    display_frame = frame.copy()

    # Draw circle at brightest spot
    if maxVal >= min_brightness:
        # Laser detected!
        cv.circle(display_frame, maxLoc, 15, (0, 255, 0), 2)
        cv.putText(display_frame, f"LASER LOCKED: {maxLoc}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw crosshair
        cv.drawMarker(display_frame, maxLoc, (0, 255, 0), cv.MARKER_CROSS, 20, 2)
    else:
        # No laser
        cv.putText(display_frame, "NO SIGNAL", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display debug info on screen
    info_text = f"Exp: {exposure_val} | Max Brightness: {int(maxVal)} | Thresh: {min_brightness}"
    cv.putText(display_frame, info_text, (10, frame.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the windows
    cv.imshow(window_name, display_frame)
    cv.imshow("Binary Mask (What Computer Sees)", thresh_mask)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()