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

    def is_hatch_targets(self, contours) -> bool:
        if len(contours) < 2:
            return False

        first_rect = cv2.minAreaRect(contours[0])
        _, (first_width, first_height), _ = first_rect
        if max(first_width, first_height) < 3:
            return False

        second_rect = cv2.minAreaRect(contours[1])
        _, (second_width, second_height), _ = second_rect
        if max(second_width, second_height) < 3:
            return False

        return True

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

    def calculate_offset(self, contours):
        first_centroid = self.calculate_centroid(contours[0])
        second_centroid = self.calculate_centroid(contours[1])

        if first_centroid[0] == -1 or second_centroid[0] == -1:
            return -1

        width = abs(first_centroid[0] - second_centroid[0])
        centroid = 0.5 * (first_centroid[0] + second_centroid[0])
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
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            if self.is_hatch_targets(contours):
                self.draw_contour(imghsv, contours[0])
                self.draw_contour(imghsv, contours[1])

                offset = self.calculate_offset(contours)

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
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if self.is_hatch_targets(contours):
            offset = self.calculate_offset(contours)
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
