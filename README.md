The HOG model.py performs detection and tracking of people in a video stream using a HOG descriptor and SVM. At each frame, it:
1. Detects people using hog.detectMultiScale.
2. Creates or updates information about people in the people dictionary, including their position, movement speed, and height.
3. Groups people by proximity using Euclidean distance, and stores group data in a file.
4. Displays rectangles around the detected people, their ID, speed and estimated height.
5. Writes information about people and groups to text files persons.txt and groups.txt.

The program also counts the total number of detected people and groups with a certain number of participants.

rthd.py is designed to detect people in an image using a HOG descriptor and an SVM classifier. The main stages of the work:
1. A pre-trained detector is used to detect people cv2.HOGDescriptor_getDefaultPeopleDetector.
2. The image is being read from the file 2.jpg using cv2.imread.
3. The hog.detectMultiScale method finds rectangular areas where people are supposed to be, with a window step (3, 3).
4. If 4 or more objects are found, the message "Too many people" is displayed in the upper left corner of the image.
5. Each detected person is surrounded by a red rectangle. The result is displayed in the window using cv2.imshow.
6. The program waits for any key to be pressed to close the window and shut down.

![humans](https://github.com/user-attachments/assets/d68c7af3-616c-4f86-91fa-6ed10062a907)
