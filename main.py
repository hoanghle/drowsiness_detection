import ultralytics 
from ultralytics import YOLO
import cv2;
from pygame import mixer
import time
from play_sound import play_alert
#print(ultralytics.__version__)  # in version yolo

model = YOLO("train3/best.pt")

mixer.init()
mixer.music.load("sound/1126.wav")
# mixer.music.play()
frame_skip = 3  #quy ước số frame ko xử lí để đỡ tốn tài nguyên
frame_count = 0     # (?) reset lại frame_count sau mỗi số awake_count đạt yêu cầu?
sleep_count = 0    #quy ước số frame được tính là ngủ để phát cảnh báo
awake_count = 0     #quy ước tỉnh liên tục bn frame mới được tính là tỉnh
sleep_threshold = frame_skip*30    #quy ước số frame tính là đang sleep
awake_sleep_threshold = frame_skip*10

alert_start_time = None

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    if frame_count % frame_skip == 0:
        results = model.predict(source=frame, save=False, show=False, conf=0.6)
        annotated_frame = results[0].plot()

        sleeping_flag = False

        for box in results[0].boxes:
            label = box.cls
            name = results[0].names[int(label)]  # Lấy tên nhãn từ id
            if name.lower() == "sleep":  # Nhãn "sleeping"
                sleeping_flag = True
                break  # Chỉ cần biết có "sleeping" là đủ
            # else:
            #     break
        if sleeping_flag:
            sleep_count += 1
            awake_count = 0  # Reset awake_count khi phát hiện "sleeping"
        else:
            awake_count += 1
            sleep_count = 0 
        
        
        if sleep_count >= sleep_threshold:
            if not mixer.music.get_busy():  # Chỉ phát nếu chưa có âm thanh
                mixer.music.play()
                alert_start_time = time.time()  # Lưu thời gian bắt đầu phát cảnh báo
            #sleep_count = 0  

        #set duration phát nhạc
        if alert_start_time and time.time() - alert_start_time >= 34:
            mixer.music.stop()
            alert_start_time = None

        if alert_start_time:
            cv2.putText( annotated_frame, "Sleeping", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,  1,  (0, 0, 255), 2, cv2.LINE_AA)

    # else:
    #     annotated_frame = frame  #không hiển thị các frame không xử lý cho trông output mượt hơn

    

    frame_count += 1

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
