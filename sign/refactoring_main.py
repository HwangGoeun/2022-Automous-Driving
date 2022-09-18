import cv2
import numpy as np

from imutils import paths
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier

####################################################################################
## hsv 임계값 지정
HSV_RED_LOWER = np.array([0, 100, 100])
HSV_RED_UPPER = np.array([10, 255, 255])
HSV_RED_LOWER1 = np.array([160, 100, 100])
HSV_RED_UPPER1 = np.array([179, 255, 255])

HSV_YELLOW_LOWER = np.array([10, 80, 120])
HSV_YELLOW_UPPER = np.array([40, 255, 255])

HSV_BLUE_LOWER = np.array([80, 160, 65])
HSV_BLUE_UPPER = np.array([140, 255, 255])

####################################################################################
## 표지판 인식
total = 0
A1 = 0
A2 = 0
A3 = 0
B1 = 0
B2 = 0
B3 = 0

####################################################################################
## 폴더 안의 이미지 분류
# initialize the data matrix and labels
print ("[INFO] extracting features...")
data = []
labels = []

# loop over the image paths in the training set
for imagePath in paths.list_images(".\\sign\\"):
	# extract the make of the car
	make = imagePath.split("\\")[-2]

	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # extract the logo of the car and resize it to a canonical width and height
	logo = cv2.resize(gray, (128, 128))

	# extract Histogram of Oriented Gradients from the logo
	H = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

	# update the data and labels
	data.append(H)
	labels.append(make)


####################################################################################
## 이미지 분류 모델 설정
# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)

model.fit(data, labels)
print(model)
print("[INFO] evaluating...")


####################################################################################
## 영상 처리 시작
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        print("Video done")
        break
    
#    img = cv2.resize(300, 300)
    cv2.imshow("original", img)
    
    # 테두리 검출
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    redBinary = cv2.inRange(hsv, HSV_RED_LOWER, HSV_RED_UPPER)	
    redBinary1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    redBinary = cv2.bitwise_or(redBinary, redBinary1)
    yellowBinary = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    blueBinary = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    binary = cv2.bitwise_or( blueBinary, (redBinary))
    contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        binary = cv2.drawContours(binary, [cnt], -1, (255,255,255), -1)

    goodContours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.bitwise_and(binary, gray)

    for cnt in goodContours:
        area = cv2.contourArea(cnt)
        if area > 2000.0:
            x, y, w, h = cv2.boundingRect(cnt)
            rate = w / h
            print("x, y, w, h")
            print(x, y, w, h)
            # 컨투어를 그린 사각형의 비율이 일정 범위 이내일 때 (너~무 직사각형)
            if rate > 0.8 and rate < 1.2:
                cv2.rectangle(img, (x, y), (x+w, y+h), (200, 152, 50), 2)
                inputImage = gray[y:y+h, x:x+w]
                logo = cv2.resize(inputImage, (128, 128))   # 판단 할 표지판의 이미지

                # hog 알고리즘 처리(비교)
                (H, hogImage) = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12), \
					cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualize=True)
                cv2.imshow("hog", hogImage)
                pred = model.predict(H.reshape(1, -1))[0]
                cv2.putText(img, pred.title(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, \
					(70, 255, 0), 2)                

####################################################################################
## 표지판 인식
                total += 1
                if pred.title() == "A1":
                    A1 += 1
                if pred.title() == "A2":
                    A2 += 1
                if pred.title() == "A3":
                    A3 += 1
                if pred.title() == "B1":
                    B1 += 1
                if pred.title() == "B2":
                    B2 += 1
                if pred.title() == "B3":
                    B3 += 1

    cv2.imshow("candidates", img)
    if cv2.waitKey(1) == 27:
        break

print(total, A1, A2, A3, B1, B2, B3)