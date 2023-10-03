import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mtcnn.mtcnn import MTCNN
import numpy as np


# создаем метод, чтобы он прорисовывал линии нобнаруженных точках
def draw_image_with_boxes(filename, result_list):
    # загрузить фотографию
    data = pyplot.imread(filename)
    # загрузить фото на график
    pyplot.imshow(data)
    # получение содержимого для прорисовки линий и прямоугольников
    ax = pyplot.gca()
    # просмотреть все лица
    for result in result_list:
        # получить координаты лиц
        x, y, width, height = result['box']
        # создание прямоуголников по координатам лица
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # добавление их на фото
        ax.add_patch(rect)
        delta_x = result['keypoints']['right_eye'][0] - result['keypoints']['left_eye'][0]
        delta_y = result['keypoints']['right_eye'][1] - result['keypoints']['left_eye'][1]
        angle = np.arctan(delta_y / delta_x)
        delta_l = np.tan(angle)*(abs(abs(result['keypoints']['left_eye'][1]-y)-height))
        # нарисовать линии симметрии
        for key, value in result['keypoints'].items():
            if key in ['left_eye', 'right_eye', 'nose']:
                line1 = Line2D([value[0]+delta_l,value[0]-delta_l],[y,y+height], color="k")
                #line = Rectangle((value[0]+delta_l, y), 0, height, fill=False, color='red')
                ax.add_line(line1)
        print("Расстояние локальных от центра:", abs(result['keypoints']['left_eye'][0] - result['keypoints']['nose'][0]),';',abs(result['keypoints']['right_eye'][0] - result['keypoints']['nose'][0]))
    # показать итог
    pyplot.show()



filename = 'venv/img/img.png'
# загрузить фотографию
pixels = cv2.imread(filename)
# Инициализировать mtcnn
detector = MTCNN()
# используя mtcnn получить координаты лиц и положение фраментов лица
faces = detector.detect_faces(pixels)
# нарисовать линии симметрии
draw_image_with_boxes(filename, faces)