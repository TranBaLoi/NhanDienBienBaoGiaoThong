import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
#############################################
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("No GPU detected, running on CPU.")

threshold = 0.8 #THRESHOLD của Xác Suất
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Chiều rộng cửa sổ
cap.set(4, 480) # Chiều dài cửa sổ
cap.set(10, 180) # Độ sáng
# IMPORT TRAINED MODEL
model = load_model('model.h5')

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Gioi han toc do 20 km/h'
    elif classNo == 1:
        return 'Gioi han toc do 30 km/h'
    elif classNo == 2:
        return 'Gioi han toc do 50 km/h'
    elif classNo == 3:
        return 'Gioi han toc do 60 km/h'
    elif classNo == 4:
        return 'Gioi han toc do 70 km/h'
    elif classNo == 5:
        return 'Gioi han toc do 80 km/h'
    elif classNo == 6:
        return 'Ket thuc gioi han toc do 80 km/h'
    elif classNo == 7:
        return 'Gioi han toc do 100 km/h'
    elif classNo == 8:
        return 'Gioi han toc do 120 km/h'
    elif classNo == 9:
        return 'Cam vuot'
    elif classNo == 10:
        return 'Cam vuot doi voi xe tren 3.5 tan'
    elif classNo == 11:
        return 'Uu tien tai nga tu ke tiep'
    elif classNo == 12:
        return 'Duong uu tien'
    elif classNo == 13:
        return 'Nhuong duong'
    elif classNo == 14:
        return 'Dung lai'
    elif classNo == 15:
        return 'Cam tat ca phuong tien'
    elif classNo == 16:
        return 'Cam phuong tien tren 3.5 tan'
    elif classNo == 17:
        return 'Cam vao'
    elif classNo == 18:
        return 'Canh bao chung'
    elif classNo == 19:
        return 'Khuc cua nguy hiem ben trai'
    elif classNo == 20:
        return 'Khuc cua nguy hiem ben phai'
    elif classNo == 21:
        return 'Khuc cua kep'
    elif classNo == 22:
        return 'Duong xoc'
    elif classNo == 23:
        return 'Duong tron'
    elif classNo == 24:
        return 'Duong hep ben phai'
    elif classNo == 25:
        return 'Cong truong'
    elif classNo == 26:
        return 'Den giao thong'
    elif classNo == 27:
        return 'Nguoi di bo'
    elif classNo == 28:
        return 'Tre em qua duong'
    elif classNo == 29:
        return 'Xe dap qua duong'
    elif classNo == 30:
        return 'Canh bao bang tuyet'
    elif classNo == 31:
        return 'Dong vat hoang da qua duong'
    elif classNo == 32:
        return 'Ket thuc tat ca gioi han toc do va cam vuot'
    elif classNo == 33:
        return 'Re phai phia truoc'
    elif classNo == 34:
        return 'Re trai phia truoc'
    elif classNo == 35:
        return 'Chi di thang'
    elif classNo == 36:
        return 'Di thang hoac re phai'
    elif classNo == 37:
        return 'Di thang hoac re trai'
    elif classNo == 38:
        return 'Giua ben phai'
    elif classNo == 39:
        return 'Giua ben trai'
    elif classNo == 40:
        return 'Vong xuyen bat buoc'
    elif classNo == 41:
        return 'Ket thuc cam vuot'
    elif classNo == 42:
        return 'Ket thuc cam vuot doi voi xe tren 3.5 tan'
while True:
    # Đọc ảnh từ Webcame
    success, imgOrignal = cap.read()

    # Xử lý ảnh
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    # cv2.putText(imgOrignal, "NHAN DANG: ", (20, 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Tiến hành dự đoán kết quả
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        print(getCalssName(classIndex), classIndex)
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35),
                font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + " %", (180, 75),
                font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break