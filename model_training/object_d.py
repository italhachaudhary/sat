from ultralytics import YOLO

if __name__ == '__main__':

    #load YOLOv8
    model = YOLO("C:\\Users\\chtal\\Desktop\\SAT\\model_training\\yolov8s.pt") #you can use specific model

    #train model
    model.train(
        data = "C:\\Users\\chtal\\Desktop\\SAT\\model_training\\data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        device='cpu', 
        pretrained=False, #Training from the scratch
        workers=4
    )

    #model evaluation
    metrics = model.val()
    print("Evaluation Merics:", metrics)

    model.predict(
        source= r"C:\Users\chtal\Desktop\SAT\model_training\test\images",
        save=True
    )