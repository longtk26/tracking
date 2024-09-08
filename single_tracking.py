import cv2
import sys
from random import randint
import time


tracker_types = ['BOOSTING', "MIL", 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

tracker_type = tracker_types[6]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()

video = cv2.VideoCapture("videos/race.mp4")

if not video.isOpened():
    print("Err loading video")
    sys.exit()

ok, frame = video.read()

if not ok:
    print("Err reading video")
    sys.exit()

# (x_top_left, y_top_left, width, hight) 
bbox = cv2.selectROI(frame) 

print(bbox)

ok = tracker.init(frame, bbox)

colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    time.sleep(0.05)

    if not ok:
        break
    ok, bbox = tracker.update(frame)
    
    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)
    
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break