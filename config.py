from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT

appconf: dict = {
    "VIDEO_FOLDER": "videos",
    "OUT_VIDEO_FOLDER": "out",
    "DATA_FOLDER": "data",
    "TEMPLATE_NAME": "templates\\DIFF_frame0_64x64_v4.jpg",
    "VIDEO_NAME": "DroneInFireForest.mp4",
    "MIN_CONF": 0.9,
    "PSR_SETTINGS": { "radius": 7 },
    "PYRAMID_TM_SETTINGS": {
        "numScales": 5, "scaleStep": 0.05, "epsilon": 1e-7
    },
    "FIND_TRANSFORMATION": {
        "gf2tParams": {
            "maxCorners": 2000, "qualityLevel": 0.05,
            "minDistance": 3, "blockSize": 10
        },
        "lkParams": {
            "winSize": (11, 11), "maxLevel": 4,
            "criteria": (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 30, 0.01),
            "minEigThreshold": 0.01
        },
    }
}