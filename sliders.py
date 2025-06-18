import cv2
from session_params import MaskSettings

import json
class MaskSettings:
    def __init__(self):
        
        self.lower = 3
        self.upper = 90
        self.open = 4
        self.close = 6
        self.x = 180
        self.y = 180
        self.width = 240
        self.height = 240
        self.region = 52
        self.minimum_area = 0.05

        self.lower_vel = 3
        self.upper_vel = 90
        self.open_vel = 4
        self.close_vel = 6

    def load_settings(self, path):
        # a really unsophisticated way of loading settings from a file
        with open(path, "r") as f:
            data = json.load(f)
        self.lower, self.upper = data["lower"], data["upper"]
        self.open, self.close = data["open"], data["close"]

        self.x, self.y = data["x"], data["y"]
        self.width, self.height = data["width"], data["height"]
        self.region = data["region"]

        self.lower_vel, self.upper_vel = data["lower_vel"], data["upper_vel"]
        self.open_vel, self.close_vel = data["open_vel"], data["close_vel"]

    def save_settings(self, path):
        data = {"lower" : self.lower,
                "upper" : self.upper,
                "open" : self.open,
                "close" : self.close,
                "lower_vel": self.lower_vel,
                "upper_vel": self.upper_vel,
                "open_vel": self.open_vel,
                "close_vel": self.close_vel,
                "x" : self.x,
                "y" : self.y,
                "width" : self.width,
                "height": self.height,
                "region": self.region}

        with open(path, 'w') as f:
            json.dump(data, f)

class Sliders:
    def nothing(self, x): pass
    def __init__(self, image_width, image_height, reference_settings_json = None):
        windowName = "Raw Mask"
        cv2.namedWindow('Raw Mask') #, cv2.WINDOW_NORMAL)
        # cv2.imshow("Sliders", np.zeros((10, 100)))
        self.image_width = image_width
        self.image_height = image_height
        self.maskSettings = MaskSettings()
        if reference_settings_json is not None:
            print("loading settings from ", reference_settings_json)
            self.maskSettings.load_settings(reference_settings_json)
        self.windowName = windowName
        cv2.createTrackbar('H_Lower', windowName, self.maskSettings.x, image_width, self.nothing)
        cv2.createTrackbar('S_Lower', windowName, self.maskSettings.y, image_height, self.nothing)
        cv2.createTrackbar('V_Lower', windowName, self.maskSettings.height, image_height, self.nothing)
        cv2.createTrackbar('Width', windowName, self.maskSettings.width, image_width, self.nothing)
        cv2.createTrackbar('Lower', windowName, self.maskSettings.lower, 255, self.nothing)
        cv2.createTrackbar('Upper', windowName, self.maskSettings.upper, 255, self.nothing)
        cv2.createTrackbar('Open', windowName, self.maskSettings.open, 10, self.nothing)
        cv2.createTrackbar('Close', windowName, self.maskSettings.close, 10, self.nothing)
        cv2.createTrackbar('Region', windowName, self.maskSettings.region, 150, self.nothing)
        cv2.createTrackbar('MinArea%', windowName, 5, 100, self.nothing)

        cv2.createTrackbar('Lower_Vel', windowName, self.maskSettings.lower_vel, 255, self.nothing)
        cv2.createTrackbar('Upper_Vel', windowName, self.maskSettings.upper_vel, 255, self.nothing)
        cv2.createTrackbar('Open_Vel', windowName, self.maskSettings.open_vel, 10, self.nothing)
        cv2.createTrackbar('Close_Vel', windowName, self.maskSettings.close_vel, 10, self.nothing)


    def updateMaskSettings(self):
        self.maskSettings.open = cv2.getTrackbarPos('Open', self.windowName)
        self.maskSettings.close = cv2.getTrackbarPos('Close', self.windowName)
        self.maskSettings.lower = cv2.getTrackbarPos('Lower', self.windowName)
        self.maskSettings.upper = cv2.getTrackbarPos('Upper', self.windowName)
        self.maskSettings.x = cv2.getTrackbarPos('X', self.windowName)
        self.maskSettings.y = cv2.getTrackbarPos('Y', self.windowName)
        self.maskSettings.height = cv2.getTrackbarPos('Height', self.windowName)
        self.maskSettings.width = cv2.getTrackbarPos('Width', self.windowName)
        self.maskSettings.region = max(1, cv2.getTrackbarPos("Region", self.windowName)) # prevents crashes from div by zero error
        self.maskSettings.minimum_area = cv2.getTrackbarPos("MinArea%", self.windowName) / 100

        self.maskSettings.open_vel = cv2.getTrackbarPos('Open_Vel', self.windowName)
        self.maskSettings.close_vel = cv2.getTrackbarPos('Close_Vel', self.windowName)
        self.maskSettings.lower_vel = cv2.getTrackbarPos('Lower_Vel', self.windowName)
        self.maskSettings.upper_vel = cv2.getTrackbarPos('Upper_Vel', self.windowName)

    # this is to work with the mouse adjustment
    def setROISettings(self, x, y, height, width):
        cv2.setTrackbarPos('X', self.windowName, x)
        cv2.setTrackbarPos('Y', self.windowName, y)
        cv2.setTrackbarPos('Height', self.windowName, height)
        cv2.setTrackbarPos('Width', self.windowName, width)

    def getSettings(self):
        self.updateMaskSettings()
        return self.maskSettings