import cv2
from ultralytics import YOLO
from ultralytics import solutions

#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 1: usb camera 0: default cameara
#size = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
model = YOLO('./model/yolov8n.pt') #load model

def Predict():
    ret, frame = capture.read() # capture 1 frame
    #frame = cv2.flip(frame, 1) # flip frame
    results = model.predict(frame, conf=0.4)
    for r in results:
        boxes = r.boxes # get bounding boxes
    num = len(boxes)
    #class_ids = boxes.cls.cpu().numpy().astype(int) # transfer to int type
    #num = np.sum(class_ids == 0)
    annotated_frame = results[0].plot() 
    return annotated_frame, num #return predicted results and num of bounding boxes

def count(times):
    equal = True
    person_calibrate = []
    for i in range(times):
        annotated_frame, num = Predict()
        if num >= 2: num = 2
        cv2.imshow("YOLO Inference", annotated_frame)
        if cv2.waitKey(50) == 27: # exit with Esc
            break
        person_calibrate.append(num)
        print(person_calibrate)
        if len(person_calibrate) == 1:
            continue
        if person_calibrate[-1] != person_calibrate[-2]:
            equal = False
            break
    if equal == False:
        count(times)
     
    return person_calibrate[-1]
        

person_count = count(10)
print(person_count)
capture.release()
cv2.destroyAllWindows()
