import cv2
import numpy

img = cv2.imread("venv/img/3.png")
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("venv/Tamples/22.png", 0)
weight, height = template.shape[::-1]

result = cv2.matchTemplate(img_grey, template, cv2.TM_CCOEFF_NORMED)

local = numpy.where(result >= 0.7)

for part in zip(*local[::-1]):cv2.rectangle(img, part, (part[0] + weight, part[1] + height), (0, 255, 255), 1)

cv2.imshow('img', img)

cv2.waitKey()