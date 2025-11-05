import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0) #webcam

#Background subtractor
subtractor = cv.createBackgroundSubtractorMOG2()
#subtractor = cv.createBackgroundSubtractorKNN()

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

    #apply subtractor
    # White = foreground, Black = background

    fg_mask = subtractor.apply(frame)
    #ret, fg_mask = cv.threshold(fg_mask, 254, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8) #creates 3x3 grid
    fg_mask = cv.erode(fg_mask, kernel, iterations=2) #applies a 3x3 grid around each pixel, if all neighbours in grid are white then keeps the white, otherwise destroy
    #so noise removed
    fg_mask = cv.dilate(fg_mask, kernel, iterations=2) #applies 3x3 grid, if any neighbour is white it restores, main object which survived erosion is "grown back"

    contours, _ = cv.findContours(fg_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #returns a list of all points along the perimeter of shapes in frame

    for contour in contours:
        print(contour) #just pass the coordinates not hierarchy

        if cv.contourArea(contour) < 500:  #min area
            continue

        (x, y, w, h) = cv.boundingRect(contour)

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv.imshow("Frame", frame)
    cv.imshow("Foreground Mask", fg_mask) #result


    #exit when q pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#destroy windows
capture.release()
cv.destroyAllWindows()