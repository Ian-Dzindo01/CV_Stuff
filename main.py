import cv2
from tracker import *


cap = cv2.VideoCapture("traffic.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=90)

tracker = EuclideanDistTracker()

while(True):
    ret, frame = cap.read()

    # Extract region of interest
    height, width, _ = frame.shape
    roi = frame[265:, ]

    mask = object_detector.apply(roi)                   # only take the window of interest
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
        area = cv2.contourArea(cnt)
        if area > 100:
            x,y,h,w = cv2.boundingRect(cnt)
            # cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            detections.append([x,y,w,h])

    # Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
