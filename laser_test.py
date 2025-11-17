import cv2 as cv
import numpy as np
import serial
import time
import threading  #Import the threading library
import queue  #Import the queue library
import math

'''
Testing with servo laser, input pixel coordinate in input_value [x,y] list in main while loop and script will send angles to servo to aim at 

'''

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

            angle_to_send = q.get()

            if angle_to_send is None:
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


    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)


    norm_point = cv.undistortPoints(point, camera_matrix, dist_coeffs, P=None)


    x_prime = norm_point[0][0][0]
    y_prime = norm_point[0][0][1]


    angle_x_rad = math.atan(x_prime)
    angle_y_rad = math.atan(y_prime)


    angle_x_deg = math.degrees(angle_x_rad)
    angle_y_deg = math.degrees(angle_y_rad)

    return angle_x_deg, angle_y_deg




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



try:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break


        #input_value = [3.229873787700622074e+02, 2.321190112338632616e+02]      #centre values

        input_value = [500,500]     #pixel coordinate for laser to point at

        centre_value_x = input_value[0]
        centre_value_y = input_value[1]

        frame_height = frame.shape[0]

        cv.circle(frame, (int(centre_value_x), int(centre_value_y)), 5, (0, 255, 0), -3)

        cv.line(frame,
               pt1=(int(centre_value_x), 0),
               pt2=(int(centre_value_x), frame_height),
               color=(0, 0, 255),
               thickness=2)

        cam_pan_angle, cam_tilt_angle = get_true_angles(centre_value_x, centre_value_y, K, D)   #calculate normalised angles

        servo_pan = 90 - cam_pan_angle


        #send method
        #put the angle in the queue.
        #non-blocking and instant (no buffering)
        command_queue.put(int(servo_pan))

        cv.imshow("Frame", frame)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Main script: Exiting...")
    #send None signal to tell the thread to exit
    command_queue.put(None)
    serial_thread.join()  #wait for thread to finish
    capture.release()
    cv.destroyAllWindows()