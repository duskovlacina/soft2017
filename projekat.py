import numpy as np
import cv2
from network import loadModel
import math


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def checkDistance(x,y):
    for broj in numbers:
        distance = math.sqrt(math.pow(x-broj.x,2) + math.pow(y-broj.y,2))
        if distance < 22:
            return broj
    return None

class Number:

    def __init__(self,x,y,added):
        self.x = x
        self.y = y
        self.added = added

def select_roi(image_orig, image_bin,lines):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    for x1, y1, x2, y2 in lines[0]:
        downLeft = x1
        UpLeft= x2
        downRight = y1
        UpRight = y2

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 10 and h < 48 and h > 10 and w > 4:
            broj = checkDistance(x,y)
            if broj is None:
                broj = Number(x, y, False)
                numbers.append(broj)
            else:
                broj.x = x
                broj.y = y
            if x >= downLeft and x<= UpLeft and y <= downRight and y >= UpRight:  #pravi kvadrat, dijagonala mu je linija
                if y - downRight >= ((UpRight - downRight) / (UpLeft - downLeft)) * (x - downLeft):
                    if broj.added is False:
                        broj.added = True
                        region = image_bin[y:y + h , x:x + w]
                        regions_array.append([resize_region(region), (x, y, w, h)])
                        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # print('xx='+ UpLeft + '  yy=0' + UpRight)
                # if xx >= UpLeft and xx <= upRight and yy <= y1 and yy >= upRight:
                # if (UpLeft+100 >= xx >= UpLeft or x2+100 >= xx >= x2) and (downRight+100 >= yy >= downRight or y2+100 >= yy >= upRight) :
                #if p > 12 and xx < 220 and yy < 220 and w >= 7 and h > 8:

    return image_orig, regions_array

def HoughLinesTransf(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    imgts = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)[1]
    minLineLength = 100
    maxLineGap = 100
    lines = cv2.HoughLinesP(imgts, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    return img,edges,lines

numbers =[]
#ucitavanje modela
model = loadModel()
cap = cv2.VideoCapture("dataset/video-2.avi")
ret, frame = cap.read()
b, g, r = cv2.split(frame)
kernel = np.ones((4, 4), np.uint8)
blur = cv2.GaussianBlur(b, (7, 7), 0)
erosion = cv2.erode(blur, kernel, iterations=1)


picture,kontura,lines = HoughLinesTransf(erosion)
cv2.imwrite('houghlines5.jpg', erosion)
konacno = 0
while (True):
    ret, frame = cap.read()
    if ret is not True :
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    imgb = image_bin(gray)

    picture, region = select_roi(frame, imgb,lines)
    for r in region:
        broj = r[0].reshape(1,1,28,28)
        maxPrediction = 0
        number = 0
        count = 0
        predikcija = model.predict(broj)
        for activation in predikcija[0]:
            if activation > maxPrediction :
                maxPrediction = activation
                number = count
            count=count+1
        konacno = konacno + number


    cv2.imshow('frame', frame)
    print('Zbir prepoznatih brojeva je ' + konacno.__str__())

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()