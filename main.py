import os
import time as tm
import traceback as tracebk

import cv2 as cv
import numpy as np
import logging as log

from config import appconf as conf
from utils import getVideoCaptore, getVideoWriter, loadTemplate, printImg


ROI_ALPHA = 0.5 # Коэффициент расширения ROI


class Bbox:
    """ Класс ограничивающей объект рамки. """
    
    def __init__(self):
        self.isInitialized = False

    def initialize(self, dimensions):
        self.xObj, self.yObj, self.wObj, self.hObj = dimensions
        self.xRoi, self.yRoi, self.wRoi, self.hRoi = (
            int(self.xObj - ROI_ALPHA*self.wObj), int(self.yObj - ROI_ALPHA*self.hObj),
            int(self.wObj + 2*ROI_ALPHA*self.wObj), int(self.hObj + 2*ROI_ALPHA*self.hObj)
        )
        self.center = (self.xObj + self.wObj//2, self.yObj + self.hObj//2)
        self.isInitialized = True
    
    def update(self, newTopLeft, templateShape):
        self.wObj, self.hObj = templateShape[1], templateShape[0]
        self.xObj, self.yObj = newTopLeft
        self.xRoi, self.yRoi, self.wRoi, self.hRoi = (
            int(self.xObj - ROI_ALPHA*self.wObj), int(self.yObj - ROI_ALPHA*self.hObj),
            int(self.wObj + 2*ROI_ALPHA*self.wObj), int(self.hObj + 2*ROI_ALPHA*self.hObj)
        )
        self.center = (self.xObj + self.wObj//2, self.yObj + self.hObj//2)

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

    mid = (1 + conf["PYRAMID_TM_SETTINGS"]["numScales"]) // 2
    scales = np.linspace(1 - conf["PYRAMID_TM_SETTINGS"]["scaleStep"]*(mid - 1),
                1 + conf["PYRAMID_TM_SETTINGS"]["scaleStep"]*(mid - 1),
                conf["PYRAMID_TM_SETTINGS"]["numScales"])
    
    scaledTplt = None
    bestMatchMap, bestVal, bestScale = None, -1.0, 1.0
    for s in scales:
        if np.round(s) == 1: # Никак не масштабируем
            scaledTplt = template
        elif s < 1.0: # Уменьшаем масштаб используя интерполяцию AREA
            scaledTplt = cv.resize(template, (0, 0), fx=s, fy=s, interpolation=cv.INTER_AREA)
        else: # Увеличиваем масштаб используя интерполяцию LINEAR
            scaledTplt = cv.resize(template, (0, 0), fx=s, fy=s, interpolation=cv.INTER_LINEAR)

        
        map = cv.matchTemplate(roi, scaledTplt, cv.TM_CCOEFF_NORMED)
        maxVal = np.max(map)
        if maxVal > bestVal:
            bestMatchMap = map
            bestVal = maxVal
            bestScale = s
    
    log.debug(f"Лучший масштаб: {bestScale:.3f}, уверенность: {bestVal:.3f}")
    return bestMatchMap, bestScale if bestScale != 1.0 else None


def findTemplate(img, template, bbox, frame):
    """ Поиск шаблона на изображении. """
    
    if not bbox.isInitialized:

        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
        bbox.initialize((maxLoc[0], maxLoc[1], template.shape[1], template.shape[0]))
        log.info(f"Инициализация рамки: {bbox.objAsTuple()}")
        return True, bbox, template

    # Вырезаем ROI из кадра
    xRoi, yRoi, wRoi, hRoi = bbox.roiAsTuple()
    roi = img[max(0, yRoi):min(img.shape[0], yRoi + hRoi), max(0, xRoi):min(img.shape[1], xRoi + wRoi)]

    # Поиск по сетке масштабов, если confidence меньше порога
    scale = None
    grayMap = cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
    _, maxVal, _, maxLoc = cv.minMaxLoc(grayMap)
    
    if maxVal < np.float32(0.89):
        log.debug(f"TM confidence {maxVal:.3f} меньше порога {0.9}, поиск по сетке масштабов...")
        grayMap, scale = multiscaleTMPyramid(roi, template)
    

    if scale is not None:
        template = cv.resize(template, (0, 0), fx=scale, fy=scale, 
            interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR)

    # Данные по лучшему совпадению и дополнительные метрики его качества
    maxLoc, conf, PSR, APCE = getAdditionalTMConfidenceMetrics(grayMap, template.shape[1], template.shape[0])
    log.debug(f"TM confidence: {conf:.3f}, PSR: {PSR:.3f}, APCE: {APCE:.3f} at frame: {frame}")

    if conf > 0.9 and PSR > 2.0 and APCE > 1.0:
        bbox.update((maxLoc[0] + xRoi, maxLoc[1] + yRoi), template.shape)
        return True, bbox, template

    return False, bbox, template





def runMainLoop(cap: cv.VideoCapture, out: cv.VideoWriter):
    """ Основной цикл обработки видео. """
    
    # Загрузка фиксированного шаблона
    # (Сценарий: захват в определенный момент времени объекта оператором)
    fixedTemplt = loadTemplate(os.path.join(
        conf["DATA_FOLDER"], conf["TEMPLATE_NAME"]))
    
    # Адаптивный текущий шаблон
    adaptiveTemplt = fixedTemplt.copy()

    frames = 0 # Счетчик обработанных кадров
    _, curFrame = cap.read()
    hFrm, wFrm = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

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
        wasFounded, bbox, adaptiveTemplt = findTemplate(curGrayFrame, adaptiveTemplt, bbox, frames)

        # Отрисовка результатов
        if wasFounded:
            # Рисуем рамку вокруг объекта
            cv.rectangle(curFrame, (bbox.xObj, bbox.yObj), (bbox.xObj + bbox.wObj, bbox.yObj + bbox.hObj), (0, 255, 0), 2)
            # Рисуем рамку вокруг ROI
            cv.rectangle(curFrame, (max(0, bbox.xRoi), max(0, bbox.yRoi)),
                (min(wFrm, bbox.xRoi + bbox.wRoi), min(hFrm, bbox.yRoi + bbox.hRoi)), (255, 0, 0), 2)
            # Добавляем текст с координатами центра объекта и ROI
            cv.putText(curFrame, f"Center: {bbox.center}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(curFrame, f"ROI: {(max(0, bbox.xRoi), max(0, bbox.yRoi), min(wFrm, bbox.xRoi + bbox.wRoi), min(hFrm, bbox.yRoi + bbox.hRoi))}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            log.info("Объект не найден на кадре.")

        cv.imshow('Video Playback', curFrame)
        # tm.sleep(0.1)
        curFrame = nxtFrame

        if frames % 100 == 0:
            log.info(f"Количество обработанных кадров: {frames}")
        
        out.write(curFrame)
        frames += 1

        # Выход по нажатию клавиши 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    log.info("Основной цикл обработки видео завершился. Количество обработанных кадров: " + str(frames))


def main():

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

        # Объект из которого читаются кадры
        cap = getVideoCaptore(os.path.join(conf["VIDEO_FOLDER"], conf["VIDEO_NAME"]))
        # Объект, создающий результирующую видеодорожку
        out = getVideoWriter("output.mp4", os.path.join(conf["DATA_FOLDER"], conf["OUT_VIDEO_FOLDER"]),
            15.0, int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        
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
