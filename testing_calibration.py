import cv2 as cv
import numpy as np
import serial
import time
import threading  #Import the threading library
import queue  #Import the queue library
import math



K = np.array([[5.502101709080033061e+02, 0, 3.229873787700622074e+02],
                    [0, 5.509288296445998867e+02, 2.321190112338632616e+02],
                    [0, 0, 1]], dtype=np.float32)
D = np.array([4.391641351825171374e-02, -2.803020736590210449e-01,-1.217892293841370006e-03, -1.528847043158600670e-03, 3.432285588703486434e-01], dtype=np.float32)

#serial worker function
# This code will run on a separate thread# Its only job is to send data to the Pico
def serial_worker(port, baud, q):
    try:
        pico = serial.Serial(port, baud, timeout=1)
        print("Serial Thread: Connected!")
        last_angle_sent = -1

        while True:
            # Wait for a new angle to arrive in the queue
            # This 'get()' command blocks, but it only
            # blocks the serial thread, NOT your main video!
            angle_to_send = q.get()

            if angle_to_send is None:  # Signal to exit
                break

            if angle_to_send != last_angle_sent:
                command = f"{angle_to_send}\n"
                pico.write(command.encode())
                last_angle_sent = angle_to_send

        pico.close()
        print("Serial Thread: Closed.")
    except serial.SerialException as e:
        print(f"Serial Thread Error: {e}")


def get_true_angles(pixel_x, pixel_y, camera_matrix, dist_coeffs):
    """
    Converts a raw pixel coordinate into horizontal and vertical
    angles (in degrees) from the camera's optical center.
    """

    # 1. Format the point for OpenCV
    #    undistortPoints expects a shape of (N, 1, 2)
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

    # 2. Undistort and Normalize
    #    We leave 'P' empty. This causes the function to return
    #    "Normalized Coordinates" (x', y') on the image plane.
    norm_point = cv.undistortPoints(point, camera_matrix, dist_coeffs, P=None)

    # Extract the normalized coordinates
    x_prime = norm_point[0][0][0]
    y_prime = norm_point[0][0][1]

    # 3. Calculate Angles using geometry
    #    Since x' = X/Z, theta = atan(x')
    angle_x_rad = math.atan(x_prime)
    angle_y_rad = math.atan(y_prime)

    # 4. Convert to Degrees
    angle_x_deg = math.degrees(angle_x_rad)
    angle_y_deg = math.degrees(angle_y_rad)

    return angle_x_deg, angle_y_deg


# --- MAIN SCRIPT ---

#setup communication line
PICO_PORT = 'COM3'  #serial communiucation line between script and pico microcontroller
PICO_BAUD = 115200  #increase baud rate

#queue
command_queue = queue.Queue()

#start the serial worker thread
#pass the port, baud rate, and the queue
serial_thread = threading.Thread(
    target=serial_worker,
    args=(PICO_PORT, PICO_BAUD, command_queue),
    daemon=True  #A daemon thread exits when the main script exits
)
serial_thread.start()

#background subtraction script
capture = cv.VideoCapture(0)

capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

subtractor = cv.createBackgroundSubtractorMOG2()

if not capture.isOpened():
    print("Error: Could not open camera")
else:
    # Get the resolution from the camera object
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame resolution: {width}x{height}")

previous_center = None
object_age = 0
min_age_to_draw = 3
#distance_threshold = 30

distance_threshold = 20
kernel = np.ones((5, 5), np.uint8)

try:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # --- NEW SECTION TO REMOVE *ALL* RED PIXELS ---

        # 1. Convert the BGR frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # 2. Define the range for red.
        #    Red is tricky because it wraps around 0/180 in OpenCV.
        #    We need to define two ranges to catch all reds.

        # Range 1: Lower red (e.g., 0-10)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv.inRange(hsv_frame, lower_red, upper_red)

        # Range 2: Upper red (e.g., 170-180)
        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask2 = cv.inRange(hsv_frame, lower_red, upper_red)

        # 3. Combine the two red masks
        #    This is our final mask of "all red pixels"
        red_mask = mask1 + mask2

        # 4. "Paint" black (0,0,0) over the red pixels
        #    This modifies the 'frame' in-place.
        frame[red_mask > 0] = (0, 0, 0)





        # --- END OF NEW SECTION ---

        # Now, the background subtractor will see a "clean" frame
        # with no laser dot on it.
        fg_mask = subtractor.apply(frame)

        #fg_mask = subtractor.apply(frame)
        fg_mask = cv.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv.dilate(fg_mask, kernel, iterations=1)
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if cv.contourArea(contour) < 100:
                continue
            (x, y, w, h) = cv.boundingRect(contour)
            boxes.append([x, y, w, h])

        #default angle (servo)
        servo_pan = 90
        servo_tilt = 90

        if len(boxes) == 0 or len(boxes) > 25:
            previous_center = None
            object_age = 0
        else:

            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            for (x, y, w, h) in boxes:
                area = w * h
                centre_x = x + w // 2
                centre_y = y + h // 2
                weighted_x += centre_x * area
                weighted_y += centre_y * area
                total_weight += area
            current_center = (weighted_x / total_weight, weighted_y / total_weight)

            if previous_center is None:
                object_age = 1
                previous_center = current_center
            else:
                dist_sq = (current_center[0] - previous_center[0]) ** 2 + (current_center[1] - previous_center[1]) ** 2
                if dist_sq < distance_threshold ** 2:
                    object_age += 1
                    previous_center = current_center
                else:
                    object_age = 1
                    previous_center = current_center

        if object_age >= min_age_to_draw:
            cv.circle(frame, (int(previous_center[0]), int(previous_center[1])), 20, (0, 255, 0), 3)
            centre_value_x = previous_center[0]
            centre_value_y = previous_center[1]

            cam_pan_angle, cam_tilt_angle = get_true_angles(centre_value_x, centre_value_y, K, D)   #calculate normalised angles

            servo_pan = 90 - cam_pan_angle
            servo_tilt = 90 - cam_tilt_angle

        final_pan = int(max(0, min(180, servo_pan)))
        final_tilt = int(max(0, min(180, servo_tilt)))

        #send method
        #put the angle in the queue.
        #non-blocking and instant (no buffering)
        command_queue.put(int(final_pan))



        cv.imshow("Frame", frame)
        cv.imshow("Foreground Mask", fg_mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Main script: Exiting...")
    #send None signal to tell the thread to exit
    command_queue.put(None)
    serial_thread.join()  #wait for thread to finish
    capture.release()
    cv.destroyAllWindows()