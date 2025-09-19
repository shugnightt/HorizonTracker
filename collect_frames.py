import os
import sys
import time

import cv2 as cv

from config import appconf as conf

VIDEO_NAME = conf["VIDEO_NAME"]
VIDEO_FOLDER = conf["VIDEO_FOLDER"]

path_to_video = os.path.join(os.path.dirname(__file__), f"{VIDEO_FOLDER}\\{VIDEO_NAME}")
if not os.path.exists(path_to_video):
    FileNotFoundError(f"Файл {VIDEO_NAME} в папке {VIDEO_FOLDER}\\ не найден!")
cap = cv.VideoCapture(path_to_video)

output_folder = 'data\\frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
frames_dir_path = os.path.join(os.path.dirname(__file__), output_folder)


def main():
    frame_count = 0
    while True:

        res, frame = cap.read()

        if not res:
            break

        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv.imwrite(frame_filename, frame)

        frame_count += 1


if __name__ == "__main__":
    main()