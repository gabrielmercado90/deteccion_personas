from detector import detector
import cv2
import time

detector = detector("yolov3-tiny.weights", "yolov3-tiny.cfg", "coco.names")

video = cv2.VideoCapture(0)
t0 = time.time()
f = 0
while video.isOpened():
    _, frame = video.read()
    f += 1
    if frame is not None:
        output = detector.predict(frame)
        #mostrar
        dt = time.time()-t0
        fps = f/dt
        cv2.putText(output, "FPS: "+str(round(fps,2)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imshow('capture', output) 
    
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break


video.release()
cv2.destroyAllWindows()