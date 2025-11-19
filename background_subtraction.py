import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
subtractor = cv.createBackgroundSubtractorMOG2()

'''
Holds main logic for the motion detection system used in main.py

Uses background and foreground separation through Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
foreground objects which are moving show up as a white blob on the image and a bounding box is drawn across each blob.

To reduce the effect of noise we use cv.erode to first make the centre pixel black if pixels in a 5x5 grid around the point are not all white then use cv.dilate to make centre pixel white
if one of the pixels in 5x5 grid are white to recover any lost blob from erode function.

Situationally the camera will change focus/ white balance or if camera shakes there will be many bounding boxes, to account for this 
when there are more than 25 bounding boxes drawn we do not analyse that frame,
additionally bounding boxes smaller than a certain area will also be thrown away.

We then take all the boxes and calculate the weighted average centre point (weighted by area) of all of them to find a single centre point of motion.
This centre point is only plot after the object has been on the frame for 3 consecutive frames to reduce noise affecting the result, the program determines if they are the 
same object by checking to see if they are within a radius of 30 pixels from the last point.
'''


#variables for motion tracker
previous_center = None  #stores the coordinates (x,y) of last target
object_age = 0  #how many frames in a row target has been tracked
min_age_to_draw = 3  #how many frames before we draw
distance_threshold = 30  #max distance between centre points
kernel = np.ones((5, 5), np.uint8)

fps = capture.get(cv.CAP_PROP_FPS)
width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
print(f"Video Info: {width}x{height} at {fps:.2f} FPS")

capture.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)


capture.set(cv.CAP_PROP_AUTO_WB, 1)


while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    #clean and get foreground mask
    fg_mask = subtractor.apply(frame)

    fg_mask = cv.erode(fg_mask, kernel, iterations=1) #make centre pixel black if not all pixels in 5x5 grid are white (filter noise)
    fg_mask = cv.dilate(fg_mask, kernel, iterations=1) #make centre pixel white if one of the pixels in 5x5 grid are white (recover lost blobs)

    #contours (edges)
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #collect boxes
    boxes = []
    for contour in contours:
        if cv.contourArea(contour) < 75:  #set minimum area for boxes to be considered
            continue
        (x, y, w, h) = cv.boundingRect(contour)
        boxes.append([x, y, w, h])

    #merge boxes -- don't think this is doing anything
    merged_boxes, _ = cv.groupRectangles(boxes, 0, 5)




    #if there is no motion (no boxes) or too many boxes (due to camera changing exposure) - reset the tracker
    if len(merged_boxes) == 0 or len(merged_boxes) > 25:
        previous_center = None #reset previous track coordinate
        object_age = 0 #reset age

    else:

        #use weighted average to determine the centre of the object in motion
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for (x, y, w, h) in merged_boxes:
            area = w * h  #use area as weight
            centre_x = x + w // 2   #calculate the centre of box
            centre_y = y + h // 2


            weighted_x += centre_x * area   #the greater the area of the bounding box the more effect it has on the final centre point
            weighted_y += centre_y * area
            total_weight += area

        #single weighted centre
        current_center = (weighted_x / total_weight, weighted_y / total_weight) #calculate centre point

        #track the centre
        if previous_center is None:
            #if new object increment age (start counting frames present)
            object_age = 1
            previous_center = current_center
        else:
            #Check distance from last known position
            dist_sq = (current_center[0] - previous_center[0]) ** 2 + (current_center[1] - previous_center[1]) ** 2

            if dist_sq < distance_threshold ** 2:   #same object

                object_age += 1
                previous_center = current_center
            else:   #new object or centre jumped - reset

                object_age = 1
                previous_center = current_center


    #draw if age of object >= 3
    if object_age >= min_age_to_draw:
        #plot circle on frame
        cv.circle(frame, (int(previous_center[0]), int(previous_center[1])), 20, (0, 255, 0), 3)

    cv.imshow("Frame", frame)
    cv.imshow("Foreground Mask", fg_mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv.destroyAllWindows()
