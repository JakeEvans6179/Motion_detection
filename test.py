import cv2 as cv
import numpy as np
import serial
import time
import threading  #Import the threading library
import queue  #Import the queue library


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
distance_threshold = 30

try:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break



        fg_mask = subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv.dilate(fg_mask, kernel, iterations=1)
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if cv.contourArea(contour) < 75:
                continue
            (x, y, w, h) = cv.boundingRect(contour)
            boxes.append([x, y, w, h])

        angle = 90  #default angle (servo)

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
            abs_x = frame.shape[1] // 2
            relative_x = centre_value_x - abs_x
            angle = 90 + (relative_x / abs_x) * 90

        #send method
        #put the angle in the queue.
        #non-blocking and instant (no buffering)
        command_queue.put(int(angle))

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