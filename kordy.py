import numpy as np
import cv2

#Tworzymy funkcje

coords = []


def click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN: #Deklarujemy ktory przycisk
        print(f"{x}, {y}")

        coords.append((x, y))
        # cv2.imshow("image", img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]#Niebieski ma pozycje 0 w bgr zielony 1 i czerwony 2
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        textbgr = str(blue) + " " + str(green) + " " + str(red)
        cv2.putText(img, textbgr, (x, y), font, .5, (0, 255, 255), 1)
        cv2.imshow("image", img)


cap = cv2.VideoCapture("los_angeles.mp4")
while True:
    _, img = cap.read()
    img = cv2.resize(img, (1366, 768))
    for coord in coords:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(coord[0]) + " " + str(coord[1])
        cv2.circle(img, coord, 5, (255, 255, 0), -1)
        cv2.putText(img, text, coord, font, .5, (255, 255, 0), 1)

    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click)
    cv2.waitKey(0)
