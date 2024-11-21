import cv2 
import imutils 

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
image = cv2.imread('2.jpg') 
image = imutils.resize(image, width=min(500, image.shape[1])) 
(humans, _) = hog.detectMultiScale(image, winStride=(3,3))
if len(humans)>=4:
    cv2.putText(image, f'Too many people', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
for (x, y, w, h) in humans: 
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) 
cv2.imshow("Image", image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()