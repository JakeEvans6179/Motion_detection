import cv2 as cv
import numpy as np


capture = cv.VideoCapture(0)
subtractor = cv.createBackgroundSubtractorMOG2()



tracked_objects = [] #previous list for checks [x,y,w,h, age]
distance_threshold = 40  #max distance (pixels) to match a box
min_age_to_draw = 5  #how many frames a box must exist to be drawn


#distance calculation function
def get_sq_dist(center1, center2):
    x1, y1 = center1
    x2, y2 = center2
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


# read and process frames
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    #get mask and clean
    fg_mask = subtractor.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv.dilate(fg_mask, kernel, iterations=1)

    #contours (edges)
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #collect all boxes
    boxes = []
    for contour in contours:
        if cv.contourArea(contour) < 75:
            continue
        (x, y, w, h) = cv.boundingRect(contour)
        boxes.append([x, y, w, h])

    #merge boxes -- don't think this is doing anything
    merged_boxes, _ = cv.groupRectangles(boxes, 0, 5)


    current_objects = [] #this round of bounding boxes
    boxes_to_draw = []

    #loop through all boxes detected in this current frame
    for (x, y, w, h) in merged_boxes:
        center_x = x + w // 2
        center_y = y + h // 2
        current_center = (center_x, center_y)

        found_match = False

        #check against boxes from last frame
        for i, (prev_x, prev_y, prev_w, prev_h, prev_age) in enumerate(tracked_objects):
            prev_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)

            dist = get_sq_dist(current_center, prev_center)

            if dist <= distance_threshold ** 2:
                #if matched update the object
                current_objects.append([x, y, w, h, prev_age + 1])
                found_match = True
                del tracked_objects[i]  #Remove from old list so we don't match it twice
                break

        if not found_match:
            # It's a new object. Add it with age 1.
            current_objects.append([x, y, w, h, 1])

    #if existed for certain frames draw position on frame
    for (x, y, w, h, age) in current_objects:
        if age >= min_age_to_draw:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Frame", frame)
    cv.imshow("Foreground Mask", fg_mask)


    #Whatever is left in current_objects becomes the "previous" list
    tracked_objects = current_objects #save all values to tracked objects "previous list"

    #exit when q pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#destroy windows
capture.release()
cv.destroyAllWindows()