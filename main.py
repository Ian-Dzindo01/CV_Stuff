import cv2

cap = cv2.VideoCapture("traffic.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

while(True):
    ret, frame = cap.read()
    # Extract region of interest
    height, width, _ = frame.shape
    roi = frame[265:, ]


    mask = object_detector.apply(roi)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
        area = cv2.contourArea(cnt)
        if area > 100:
            x,y,h,w = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)
            # cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)



    cv2.imshow("Frame", frame)
    # cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
