import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0) #webcam

#video properties
fps = capture.get(cv.CAP_PROP_FPS)
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
print(f"Video Info: {width}x{height} at {fps:.2f} FPS")

#read and process frames
while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        print("End of video or failed to read frame")
        break



    cv.imshow("Frame", frame)


        # exit when q pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # end script
capture.release()
cv.destroyAllWindows()
