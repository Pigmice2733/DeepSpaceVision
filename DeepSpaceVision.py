import math

import numpy as np
import cv2

import libjevois as jevois


class DeepSpaceVision:
    def __init__(self):
        self.min_threshold = np.array([55, 190, 80], dtype=np.uint8)
        self.max_threshold = np.array([90, 255, 255], dtype=np.uint8)
        self.mask_only = False

        self.percent_offset = 0.9

    def draw_rect(self, imghsv, contour):
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect)
        pts = np.int0(pts)
        cv2.drawContours(imghsv, [pts], 0, (120, 255, 255), 2)

    def draw_contour(self, imghsv, contour):
        cv2.drawContours(imghsv, [contour], 0, (120, 255, 255), 2)

    def calculate_centroid(self, contour):
        moment = cv2.moments(contour)
        if moment["m00"] != 0.0:
            return (
                int(moment["m10"] / moment["m00"]),
                int(moment["m01"] / moment["m00"]),
            )
        return -1, -1

    def find_vision_targets(self, contours):
        if len(contours) < 2:
            return False, None, None

        left_side = []
        right_side = []
        
        for c in contours:
            rect = cv2.minAreaRect(c)
            size = rect[1]
            if(max(size[0], size[1]) < 5):
                continue

            angle = rect[2]
            if(size[0] > size[1]):
                angle += 90.0

            if(abs(angle - 14.5) < 5):
                left_side.append(self.calculate_centroid(c))
            elif(abs(angle + 14.5) < 5):
                right_side.append(self.calculate_centroid(c))

        if(len(left_side) == 0 or len(right_side) == 0):
            return False, None, None

        left_side = sorted(left_side, key=lambda c: c[0], reverse=False)
        right_side = sorted(right_side, key=lambda c: c[0], reverse=True)

        left = left_side[0]
        right = right_side[0]

        for r in right_side:
            if(r[0] < right[0] and r[0] > left[0]):
                right = r

        for l in left_side:
            if(l[0] > left[0] and l[0] < right[0]):
                left = l

        return True, left, right

    def calculate_offset(self, left, right):
        width = abs(right[0] - left[0])
        centroid = 0.5 * (left[0] + right[0])
        centroid += width * self.percent_offset * 0.5

        offset = (centroid - 80.0) / 80.0

        return offset

    def process(self, inframe, outframe):
        imgbgr = inframe.getCvBGR()
        imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(imghsv, self.min_threshold, self.max_threshold)

        if not self.mask_only:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in contours:
                if(cv2.contourArea(c) > 8):
                    self.draw_contour(imghsv, c)

            detected, left, right = self.find_vision_targets(contours)

            if detected:
                offset = self.calculate_offset(left, right)

                cv2.line(
                    imghsv,
                    (int(offset * 80.0 + 80.0), int(0)),
                    (int(offset * 80.0 + 80.0), int(159)),
                    (120, 255, 255),
                )

                jevois.sendSerial("OFF {} END".format(offset))

            else:
                jevois.sendSerial("OFF -5 END")

            bgr_img = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
        else:
            mask = cv2.bitwise_and(imghsv, imghsv, mask=mask)
            bgr_img = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        outframe.sendCvBGR(bgr_img)

    def processNoUSB(self, inframe):
        imgbgr = inframe.getCvBGR()
        imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(imghsv, self.min_threshold, self.max_threshold)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)

        detected, left, right = self.find_vision_targets(contours)

        if detected:
            offset = self.calculate_offset(left, right)
            jevois.sendSerial("OFF {} END".format(offset))

        else:
            jevois.sendSerial("OFF -5 END")

    def parseSerial(self, msg: str):
        if msg.startswith("set-mask"):
            return self.set_mask_only(msg[9:])
        return "ERR: Unsupported command"

    def supportedCommands(self):
        # use \n seperator if your module supports several commands
        return "set-mask - control whether mask or normal"

    def set_mask_only(self, val: str) -> str:
        if val == "on":
            self.mask_only = True
        else:
            self.mask_only = False
        return "OK"
