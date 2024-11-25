import ultralytics 
from ultralytics import YOLO
import cv2;
#print(ultralytics.__version__)  # In ra phiên bản YOLO

model = YOLO("train3/best.pt")

img = cv2.imread("img_test/test4.jpg")

#results = model.predict(source=img, show=True)  
results = model.predict(source=img) 

scale_percent = 50  # Scale 50% so với kích thước gốc
for result in results:
    img_result = result.plot()  # Lấy ảnh kết quả từ model

    # Tính kích thước mới
    width = int(img_result.shape[1] * scale_percent / 100)
    height = int(img_result.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize hình ảnh
    scaled_result = cv2.resize(img_result, dim)

    # Hiển thị kết quả
    cv2.imshow("Scaled Result", scaled_result)
    cv2.waitKey(0)

cv2.waitKey()