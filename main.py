import ultralytics 
from ultralytics import YOLO
import cv2;
#print(ultralytics.__version__)  # In ra phiên bản YOLO

model = YOLO("train3/best.pt")

frame_skip = 3  #quy ước số frame ko xử lí để đỡ tốn tài nguyên
frame_count = 0     # (?) reset lại frame_count sau mỗi số awake_count đạt yêu cầu?
sleeping_count = 0    #quy ước số frame được tính là ngủ để phát cảnh báo
awake_count = 0     #
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    if frame_count % frame_skip == 0:
        results = model.predict(source=frame, save=False, show=False, conf=0.6)
        annotated_frame = results[0].plot()
    # else:
    #     annotated_frame = frame  #không hiển thị các frame không xử lý cho trông output mượt hơn

    frame_count += 1

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
