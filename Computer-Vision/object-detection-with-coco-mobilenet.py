import cv2

#load model and class labels
prototxt_file = "Computer-Vision\\models\\frozen_inference_graph.pb"
model_file = "Computer-Vision\\models\\frozen_inference_graph.pb"
net = cv2.dnn.readNetFromTensorflow(model_file, prototxt_file)

with open("Computer-Vision\\models\\ssd_mobilenet_v1_class.txt","r") as f:
    classes = f.read().strip().split("\n")

#hyperparameters
confidence_threshold = 0.5

#----Case 1: Car detection from a video stream (video file
vid = cv2.VideoCapture("Computer-Vision\\media\\traffic-video.mp4")
while vid.isOpened():
    ret, frame = vid.read()
    frame = cv2.resize(frame, dsize=(400, 300))
    (height, width, channels) = frame.shape
	
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    networkOutput = net.forward()
    
    for detection in networkOutput[0, 0]:
        confidence = detection[2]
        
        if confidence > confidence_threshold:
            class_index = int(detection[1])
            
            startX = detection[3] * width
            startY = detection[4] * height
            endX = detection[5] * width
            endY = detection[6] * height   
            
            if startY - 15 > 15:
                y = startY - 15
            else:
                y = startY + 15
        
            text = "{}: {:.2f}%".format(classes[class_index], confidence*100)
            cv2.rectangle(frame, pt1=(int(startX), int(startY)), pt2=(int(endX), int(endY)), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, text, (int(startX), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("Car Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

#----Case 2: Car detection from an image file
img = cv2.imread("Computer-Vision\\media\\cars.jpg")

img = cv2.resize(img, dsize=(300, 300))
(height, width, channels) = img.shape

blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
net.setInput(blob)
networkOutput = net.forward()

for detection in networkOutput[0, 0]:
    confidence = detection[2]
    
    if confidence > confidence_threshold:
        class_index = int(detection[1])
        
        startX = detection[3] * width
        startY = detection[4] * height
        endX = detection[5] * width
        endY = detection[6] * height   
        
        if startY - 15 > 15:
            y = startY - 15
        else:
            y = startY + 15
    
        text = "{}: {:.2f}%".format(classes[class_index], confidence*100)
        cv2.rectangle(img, pt1=(int(startX), int(startY)), pt2=(int(endX), int(endY)), color=(0, 255, 0), thickness=2)
        cv2.putText(img, text, (int(startX), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
    
cv2.imshow("Car Detection", img)
cv2.waitKey(0) == ord('q')
    