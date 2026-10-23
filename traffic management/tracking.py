import cv2
from ultralytics import YOLO

# Load your custom fine-tuned model (trained on cars + forklifts)
model = YOLO("yolov8n-custom.pt") 

cap = cv2.VideoCapture("traffic_SampleVideo_LowQuality.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Run detection and tracking simultaneously
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    # Extract bounding boxes, tracking IDs, and class labels
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    
    for box, track_id, cls in zip(boxes, track_ids, clss):
        label = model.names[cls]
        
        # 1. Logic here to calculate distance moved by track_id since last frame
        # 2. Convert pixels to meters using your camera's scale mapping
        # 3. Speed = distance / time_elapsed
        
        # Annotate frame
        cv2.putText(frame, f"ID: {track_id} {label} | Speed: CALC_SPEED km/h", 
                    (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
