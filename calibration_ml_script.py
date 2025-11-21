import cv2 as cv
import numpy as np
import serial
import time
import math
import csv

# --- CONFIGURATION ---
PICO_PORT = 'COM3'
PICO_BAUD = 115200
K = np.array([[5.502101709080033061e+02, 0, 3.229873787700622074e+02],
              [0, 5.509288296445998867e+02, 2.321190112338632616e+02],
              [0, 0, 1]], dtype=np.float32)
D = np.array([4.391641351825171374e-02, -2.803020736590210449e-01, -1.217892293841370006e-03, -1.528847043158600670e-03,
              3.432285588703486434e-01], dtype=np.float32)

EXPOSURE_VAL = -7 #determined from camera_testing.py as the ideal values to track laser
MIN_BRIGHTNESS = 100


#from main script
def get_true_angles(pixel_x, pixel_y, camera_matrix, dist_coeffs):
    """
    Converts a raw pixel coordinate into horizontal and vertical
    angles (in degrees) from the camera's optical center.
    """

    #format point for opencv to undistort

    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

    #undistort and normalise

    #return normalized coordinates [x', y'] on image plane.
    norm_point = cv.undistortPoints(point, camera_matrix, dist_coeffs, P=None)

    #extract normalized coordinates
    x_prime = norm_point[0][0][0]
    y_prime = norm_point[0][0][1]

    #calculate angle from principal axis

    angle_x_rad = math.atan(x_prime)
    angle_y_rad = math.atan(y_prime)

    #turn back to degrees
    angle_x_deg = math.degrees(angle_x_rad)
    angle_y_deg = math.degrees(angle_y_rad)

    return angle_x_deg, angle_y_deg


# --- SETUP ---
try:
    pico = serial.Serial(PICO_PORT, PICO_BAUD, timeout=1)
    time.sleep(2)
    print("Connected to Pico.")
except:
    print("Error: Could not connect to Pico.")
    exit()

capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
capture.set(cv.CAP_PROP_EXPOSURE, EXPOSURE_VAL)

data_points = []  #Stores [Camera_Angle, Servo_Angle]

#Scan range (Servo 40 to 140 degrees)
scan_range = range(40, 141, 1)

print("Starting Calibration Sweep...")

for servo_angle in scan_range:
    #move servo
    pico.write(f"{servo_angle}\n".encode())
    time.sleep(0.5)  #wait for servo to move

    #read frame
    ret, frame = capture.read()
    if not ret: break

    #find laser
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)

    if maxVal >= MIN_BRIGHTNESS:
        pixel_x, pixel_y = maxLoc

        #convert to camera angle
        cam_angle_deg, unused_y = get_true_angles(pixel_x, pixel_y, K, D)   #y to be used later

        print(f"Servo: {servo_angle}° -> Cam Angle: {cam_angle_deg:.2f}°")
        data_points.append([cam_angle_deg, servo_angle])

        #plot circle on screen to see
        cv.circle(frame, maxLoc, 10, (0, 255, 0), 2)
        cv.imshow("Calibration", frame)
        cv.waitKey(1)
    else:
        print(f"Servo: {servo_angle}° -> Laser lost!")

#complete
capture.release()
pico.close()
cv.destroyAllWindows()

#write to csv file
with open('angle_calibration.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Camera_Angle', 'Servo_Angle'])
    writer.writerows(data_points)

print("Data saved to angle_calibration.csv")