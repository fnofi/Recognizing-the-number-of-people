import time
import cv2
import numpy as np
from collections import defaultdict

total_human_count = 0
max_detection_delay = 30
real_object_height = 1  # The height of the fence in meters
object_height_in_pixels = 80  # The size of the fence in pixels
pixels_per_meter = object_height_in_pixels / real_object_height
cap = cv2.VideoCapture("city.mp4")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
people = {}
group_threshold = 10
groups = defaultdict(list)
persons_file = open("persons.txt", "w")
groups_file = open("groups.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (680, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    for person_id, (bbox, delay, last_position, last_timestamp) in list(people.items()):
        people[person_id] = (bbox, delay + 1, last_position, last_timestamp)

    for (x, y, w, h) in boxes:
        matched = False
        new_bbox = (x, y, x + w, y + h)

        for person_id, (bbox, delay, last_position, last_timestamp) in list(people.items()):
            if delay < max_detection_delay:
                intersect_area = max(0, min(new_bbox[2], bbox[2]) - max(new_bbox[0], bbox[0])) * max(0, min(new_bbox[3], bbox[3]) - max(new_bbox[1], bbox[1]))
                new_bbox_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                overlap = intersect_area / float(new_bbox_area + bbox_area - intersect_area)

                if overlap > 0.5:
                    matched = True
                    people[person_id] = (new_bbox, 0, (x + w/2, y + h/2), time.time())
                    break

        if not matched:
            total_human_count += 1
            people[total_human_count] = (new_bbox, 0, (x + w/2, y + h/2), time.time())

    # Defining groups
    for person_id, (bbox, delay, last_position, last_timestamp) in people.items():
        if delay < max_detection_delay:
            (x, y, x1, y1) = bbox
            person_center = (x + w/2, y + h/2)

            for group_id, group_members in groups.items():
                for member_id in group_members:
                    member_info = people.get(member_id, (None, None))
                    member_bbox = member_info[0]
                    if member_bbox is not None:
                        (member_x, member_y, member_x1, member_y1) = member_bbox
                        member_center = (member_x + (member_x1 - member_x) / 2, member_y + (member_y1 - member_y) / 2)
                        distance = np.linalg.norm(np.array(member_center) - np.array(person_center))
                        if distance < group_threshold:
                            group_members.append(person_id)
                            break
                else:
                    continue
                break
            else:
                groups[person_id] = [person_id]

    for person_id, (bbox, delay, last_position, last_timestamp) in people.items():
        if delay < max_detection_delay:
            (x, y, x1, y1) = bbox
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {person_id}', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

            # Calculation of movement speed
            if last_position is not None:
                current_time = time.time()
                time_diff = current_time - last_timestamp
                if time_diff == 0:
                    speed = 0.0
                else:
                    distance = np.linalg.norm(np.array(last_position) - np.array((x + w/2, y + h/2)))
                    speed = distance / time_diff
                    speed = speed * 3.6 / 4
                if speed!=0:
                    cv2.putText(frame, f'Speed: {speed:.2f} km/h', (x, y - 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                # Determining growth based on a large-scale relationship
                person_height_in_pixels = y1 - y
                person_height_in_meters = person_height_in_pixels / pixels_per_meter
                cv2.putText(frame, f'Height: {person_height_in_meters:.2f} meters', (x, y - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                persons_file.write(f'Person {person_id}: Speed={speed:.2f} km/h, Height={person_height_in_meters:.2f} meters, Total count: {total_human_count}\n')

    # Counting groups
    group_sizes = [len(group_members) for group_members in groups.values()]
    group_2 = group_sizes.count(2)
    group_3 = group_sizes.count(3)
    group_4 = group_sizes.count(4)
    group_5 = group_sizes.count(5)
    groups_file.write(f'Group_2={group_2}, Group_3={group_3}, Group_4={group_4}, Group_5={group_5}\n')
    cv2.putText(frame, f'Count {total_human_count}', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
persons_file.close()
groups_file.close()
