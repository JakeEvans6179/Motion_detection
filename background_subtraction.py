import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
subtractor = cv.createBackgroundSubtractorMOG2()


#variables for motion tracker
previous_center = None  #stores the coordinates (x,y) of last target
object_age = 0  #how many frames in a row target has been tracked
min_age_to_draw = 3  #how many frames before we draw
distance_threshold = 30  #max distance between centre points


while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    #clean and get foreground mask
    fg_mask = subtractor.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
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
