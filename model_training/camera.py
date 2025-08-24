import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\chtal\Desktop\SAT\model_training\runs\detect\train\weights\best.pt")

#custom lables
custom_lables={
    0: "Scalpel Handle",
    1: "Tweezers",
    2: "Straight Scissors",
    3: "Curved Scissors",

}
cap = cv2.VideoCapture(0) #(camera index 0)

if not cap.isOpened():
    print("can't open the video stream.")
    exit()

cv2.namedWindow("Surgical Assistance Tool", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't read frame from video stream")
        break

    #perform inference
    results= model.predict(source=frame,save=False,conf= 0.8)

    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        label = f"{custom_lables.get(class_id,'Unknown')} ({confidence:.2f})"

        #draw bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0,0,255),2)
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    cv2.imshow("Surgical Tool Detection",frame)

    #q for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
