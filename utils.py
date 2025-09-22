import os
import logging as log

import cv2 as cv

from config import appconf


def printImg(img, winname, printAsGray = False):
    cv.imshow(winname, cv.cvtColor(img, cv.COLOR_BGR2GRAY) if printAsGray else img)
    cv.waitKey()
    cv.destroyWindow(winname)


def getVideoCapture(passToVideo: str) -> cv.VideoCapture:
    if not os.path.exists(passToVideo):
        raise FileNotFoundError(f"Файл {passToVideo} не найден!")
    return cv.VideoCapture(passToVideo)


def getVideoWriter(filename, passToFolder, fps, width, height) -> cv.VideoWriter:
    if not os.path.exists(passToFolder):
        raise FileNotFoundError(f"Дериктория {passToFolder} не найдена!")
    # Настройка объекта recorder-а для записи полученного видео
    outputFilename = os.path.join(passToFolder, filename)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    return cv.VideoWriter(outputFilename, fourcc, fps, (width, height))


def loadTemplate(passToFile):
    template = cv.imread(passToFile, cv.IMREAD_GRAYSCALE)
    TPTH, TPTW = template.shape
    log.info(f"Шаблон успешно загружен. Размерность: {(TPTW, TPTH)}")
    return template