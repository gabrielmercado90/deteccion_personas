import cv2 
import numpy as np

class detector:
    def __init__(self, pesos, model, clases):
        #red
        self.net = cv2.dnn.readNet(pesos, model)
        with open(clases, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            self.outputlayers = self.net.getUnconnectedOutLayersNames()


    def predict(self, img):
        img2 = img.copy()
        height, width, channels = img2.shape
        blob = cv2.dnn.blobFromImage(img2, 0.00392, (320,320), (0,0,0), True, crop = False)

        self.net.setInput(blob)
        outs = self.net.forward(self.outputlayers)
        boxes = []
        confidences = []

        #show information
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0: #ONLY HUMANS
                    #person detected
                    center_x = int(detection[0]* width)
                    center_y = int(detection[1]* height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                #draw rectangle & confidence
                cv2.rectangle(img2,(x,y),(x + w,y + h),(255,0,0),2)
                cv2.putText(img2, 'Persona: '+"%.2f"%(confidence*100)+'%', (x + 4, y + 20), font, 1, (255,0,0), 1)
        return img2
        
