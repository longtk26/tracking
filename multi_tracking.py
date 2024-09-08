import cv2
import sys
from random import randint
import time


tracker_types = ['BOOSTING', "MIL", 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print("Invalid tracker algorithm!")

    return tracker

def getRandomColor():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

video = cv2.VideoCapture("videos/race.mp4")

if not video.isOpened():
    print("Err loading video")
    sys.exit()

ok, frame = video.read()

bboxes = []
colors = []

while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append(getRandomColor())
    
    print("Press any key to continue")
    k = cv2.waitKey(0) & 0xFF
    if k == 113: # Q - quit
        break

tracker_type = "CSRT"
multi_tracker = cv2.legacy.MultiTracker_create()

for bbox in bboxes:
    multi_tracker.add(create_tracker_by_name(tracker_type), frame, bbox)

while video.isOpened():
    ok, frame = video.read()

    if not ok:
        break

    ok, boxes = multi_tracker.update(frame)

    for i, new_box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in new_box]
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[i], 2)

    cv2.imshow("MultiTracker", frame)
    if cv2.waitKey(1) & 0XFF == 27: #esc
        break