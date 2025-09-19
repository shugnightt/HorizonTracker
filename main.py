import os
import time as tm
import traceback as tracebk

import cv2 as cv
import numpy as np
import logging as log

from config import appconf as conf
from utils import getVideoCaptore, getVideoWriter, loadTemplate


ROI_ALPHA = 0.5 # Коэффициент расширения ROI


class Bbox:
    """ Класс ограничивающей объект рамки. """
    
    def __init__(self):
        self.isInitialized = False

    def initialize(self, topLeft, templateShape):
        """ Инициализация рамки по заданным координатам и размерам. """

        self.xObj, self.yObj = topLeft
        self.wObj, self.hObj = templateShape[1], templateShape[0]
        self.xRoi, self.yRoi, self.wRoi, self.hRoi = (
            int(self.xObj - ROI_ALPHA*self.wObj), int(self.yObj - ROI_ALPHA * self.hObj),
            int(self.wObj + 2 * ROI_ALPHA * self.wObj), int(self.hObj + 2 * ROI_ALPHA * self.hObj)
        )
        self.center = (self.xObj + self.wObj // 2, self.yObj + self.hObj // 2)
        self.isInitialized = True
    
    def update(self, newTopLeft, templateShape):
        """ Обновление координат и размеров рамки по новым данным. """

        self.wObj, self.hObj = templateShape[1], templateShape[0]
        self.xObj, self.yObj = newTopLeft
        self.xRoi, self.yRoi, self.wRoi, self.hRoi = (
            int(self.xObj - ROI_ALPHA * self.wObj), int(self.yObj - ROI_ALPHA * self.hObj),
            int(self.wObj + 2 * ROI_ALPHA * self.wObj), int(self.hObj + 2 * ROI_ALPHA * self.hObj)
        )
        self.center = (self.xObj + self.wObj // 2, self.yObj + self.hObj // 2)

    def objAsTuple(self):
        return (self.xObj, self.yObj, self.wObj, self.hObj)
    
    def roiAsTuple(self):
        return (self.xRoi, self.yRoi, self.wRoi, self.hRoi)


def preprocessing(frame, claheResult = False, blurrResult = False):
    """ Предобработка кадра: перевод в серый, CLAHE, блюр. По умолчанию
    CLAHE и блюр не применяются. """
    
    curGrayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if claheResult:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        curGrayFrame = clahe.apply(curGrayFrame)

    if blurrResult:
        curGrayFrame = cv.GaussianBlur(curGrayFrame, (5, 5), 0)

    return curGrayFrame


def getAdditionalTMConfidenceMetrics(map, roiW, roiH):
    """ Дополнительные метрики качества найденного шаблона. """
    
    mapH, mapW = map.shape
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(map)
    maxX, maxY = maxLoc
    
    # PSR
    yCoords, xCoords = np.ogrid[:mapH, :mapW]
    distFromMaxLoc = np.sqrt(np.power(xCoords - maxX, 2) + np.power(yCoords - maxY, 2))
    mask = distFromMaxLoc > conf["PSR_SETTINGS"]["radius"]
    meanVal = np.mean(map[mask])
    stdVal = np.std(map[mask])
    PSR = (maxVal - meanVal) / stdVal if stdVal > 0 else 0.0

    # APCE
    tmp = np.mean(np.power(map - minVal, 2))
    APCE = np.power(maxVal - minVal, 2) / tmp if tmp > 0 else 0.0

    return maxLoc, maxVal, PSR, APCE


def multiscaleTMPyramid(roi, template):
    """ Поиск шаблона на изображении по сетке масштабов и выбор наиболее уверенного. """

    # Формируем сетку масштабов для последовательного поиска
    # наиболее подходящего размера шаблона с учетом изменения расстояния до объекта
    mid = (1 + conf["PYRAMID_TM_SETTINGS"]["numScales"]) // 2
    scales = np.linspace(
        1 - conf["PYRAMID_TM_SETTINGS"]["scaleStep"] * (mid - 1),
        1 + conf["PYRAMID_TM_SETTINGS"]["scaleStep"] * (mid - 1),
        conf["PYRAMID_TM_SETTINGS"]["numScales"]
    )
    
    scaledTplt = None
    bestMatchMap, bestVal, bestScale = None, -1.0, 1.0

    # Для каждого масштаба строим карту совпадений и выбираем наиболее уверенное.
    # Меняем значения переменных bestMatchMap, bestVal, bestScale только если
    # найдено более уверенное совпадение чем на предыдущих итерациях.
    for scale in scales:
        if np.abs(scale - 1.0) < conf["PYRAMID_TM_SETTINGS"]["epsilon"]: # Никак не масштабируем
            scaledTplt = template
        elif scale < 1.0: # Уменьшаем масштаб используя интерполяцию AREA
            scaledTplt = cv.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        else: # Увеличиваем масштаб используя интерполяцию LINEAR
            scaledTplt = cv.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

        
        map = cv.matchTemplate(roi, scaledTplt, cv.TM_CCOEFF_NORMED)
        maximum = np.max(map)

        if maximum > bestVal:
            bestMatchMap = map
            bestVal = maximum
            bestScale = scale

    log.debug(f"Лучший масштаб: {bestScale:.3f}, уверенность: {bestVal:.3f}")
    return bestMatchMap, bestScale if bestScale != 1.0 else None


def findTemplate(img, template, bbox: Bbox, frame):
    """ Поиск шаблона на изображении. """
    
    # Инициализация рамки при первом вызове
    # Шаблон ищется по всему кадру
    if not bbox.isInitialized:
        map = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        _, _, _, maxLoc = cv.minMaxLoc(map)
        bbox.initialize(maxLoc, template.shape)
        log.info(f"Инициализация рамки: {bbox.objAsTuple()}")
        return True, template, bbox

    # Вырезаем ROI из кадра, в котором будет производиться поиск
    # Проверка на выход за границы изображения и корректировка
    xRoi, yRoi, wRoi, hRoi = bbox.roiAsTuple()
    x0 = max(0, xRoi); y0 = max(0, yRoi)
    x1 = min(img.shape[1], xRoi + wRoi); y1 = min(img.shape[0], yRoi + hRoi)
    roi = img[y0:y1, x0:x1]

    # Поиск по сетке масштабов, если confidence меньше порога
    scale = None
    grayMap = cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv.minMaxLoc(grayMap)
    
    minConf = conf["MIN_CONF"]
    if maxVal < minConf:
        log.debug(f"TM confidence {maxVal:.3f} меньше порога {minConf}, поиск по сетке масштабов...")
        grayMap, scale = multiscaleTMPyramid(roi, template)
    

    if scale is not None:
        template = cv.resize(template, (0, 0), fx=scale, fy=scale, 
            interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR)

    # Данные по лучшему совпадению и дополнительные метрики его качества
    maxLoc, confidence, PSR, APCE = getAdditionalTMConfidenceMetrics(grayMap, template.shape[1], template.shape[0])
    log.debug(f"TM confidence: {confidence:.3f}, PSR: {PSR:.3f}, APCE: {APCE:.3f} at frame: {frame}")

    if confidence > 0.8:
        bbox.update((maxLoc[0] + x0, maxLoc[1] + y0), template.shape)
        return True, template, bbox
    else:
        """ TODO: Логика обработки случая когда объект не найден. Пока не знаю что делать."""

    return False, template, bbox





def runMainLoop(cap: cv.VideoCapture, out: cv.VideoWriter):
    """ Основной цикл обработки видео. """
    start = tm.time()

    # Загрузка фиксированного шаблона
    # (Сценарий: захват в определенный момент времени объекта оператором)
    fixedTemplt = loadTemplate(os.path.join(
        conf["DATA_FOLDER"], conf["TEMPLATE_NAME"]))
    
    # Адаптивный шаблон текущей итерации обработки
    adaptiveTemplt = fixedTemplt.copy()

    frames = 0 # Счетчик обработанных кадров
    _, curFrame = cap.read()
    frmH, frmW = curFrame.shape[0], curFrame.shape[1]

    bbox = Bbox() # Объект ограничивающей шаблон рамки

    # Цикл чтения последовательности кадров видео и захвата объекта
    while True:
        
        # Берем очередной кадр
        ret, frame = cap.read()
        nxtFrame = frame
        
        if not ret:
            break

        # Предобработка, в частности перевод в серый, блюр и CLAHE
        curGrayFrame = preprocessing(curFrame, claheResult=True, blurrResult=False)

        # Поиск шаблона на текущем кадре
        wasFounded, adaptiveTemplt, bbox = findTemplate(curGrayFrame, adaptiveTemplt, bbox, frames)

        # Отрисовка результатов
        if wasFounded:

            # Рисуем рамку вокруг объекта
            cv.rectangle(curFrame,
                (bbox.xObj, bbox.yObj),
                (bbox.xObj + bbox.wObj, bbox.yObj + bbox.hObj),
                (0, 255, 0), 2
            )
            
            # Рисуем рамку вокруг ROI
            cv.rectangle(curFrame,
                (max(0, bbox.xRoi), max(0, bbox.yRoi)),
                (min(frmW, bbox.xRoi + bbox.wRoi), min(frmH, bbox.yRoi + bbox.hRoi)),
                (255, 0, 0), 2
            )
            
            # Добавляем текст с координатами центра объекта, размеров самого объекта и ROI
            cv.putText(curFrame, f"Center: {bbox.center}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(curFrame, f"OBJ: {(max(0, bbox.xObj), max(0, bbox.yObj), bbox.wObj, bbox.hObj)}",
                (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.putText(curFrame, f"ROI: {(max(0, bbox.xRoi), max(0, bbox.yRoi), bbox.wRoi, bbox.hRoi)}",
                (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        else:
            log.info("Объект не найден на кадре.")

        cv.imshow('Video Playback', curFrame)
        out.write(curFrame)

        curFrame = nxtFrame
        frames += 1

        if frames % 100 == 0:
            log.info(f"Количество обработанных кадров: {frames}")
        
        # Выход по нажатию клавиши 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    log.info("Основной цикл обработки видео завершился. Количество обработанных кадров: " + str(frames))
    log.info(f"Обработка в среднем {frames / (tm.time() - start)} FPS")

def main():

    # Настройка логгирования
    log.basicConfig(
        level=log.DEBUG,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[
            log.StreamHandler()
        ]
    )

    cap = out = None
    try:
        # Папка проекта
        basedir = os.path.dirname(os.path.abspath(__file__))
        conf["VIDEO_FOLDER"] = os.path.join(basedir, conf["VIDEO_FOLDER"])
        log.info("\n" + basedir + "\n" + conf["VIDEO_FOLDER"])

        # Создание объекта из которого читаются кадры
        cap = getVideoCaptore(os.path.join(conf["VIDEO_FOLDER"], conf["VIDEO_NAME"]))

        # Создание объекта, записывающего результирующую видеодорожку
        out = getVideoWriter("output.mp4", os.path.join(conf["DATA_FOLDER"], conf["OUT_VIDEO_FOLDER"]),
            float(cap.get(cv.CAP_PROP_FPS)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        
        # Основной цикл тестирования трекера
        runMainLoop(cap, out)

    except Exception as error:
        log.error(error)
        tracebk.print_exc()
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv.destroyAllWindows()
        log.info("Программа завершила свою работу...")


if __name__ == "__main__":
    main()
