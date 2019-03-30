import math

import numpy as np
import cv2

import libjevois as jevois


class DeepSpaceVision:
    def __init__(self):
        self.min_threshold = np.array([55, 190, 80], dtype=np.uint8)
        self.max_threshold = np.array([90, 255, 255], dtype=np.uint8)
        self.mask_only = False

        self.percent_offset = 0.55

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

    def find_vision_targets(self, contours, img_width):
        if len(contours) < 2:
            return False, None, None
        else:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            return True, self.calculate_centroid(contours[0]), self.calculate_centroid(contours[1])
        # elif len(contours) == 2:
        #     return True, self.calculate_centroid(contours[0]), self.calculate_centroid(contours[1])

        # left_side = []
        # right_side = []

        # for c in contours:
        #     rect = cv2.minAreaRect(c)
        #     size = rect[1]
        #     if(max(size[0], size[1]) < 3):
        #         continue

        #     angle = rect[2]
        #     if(size[0] > size[1]):
        #         angle += 90.0

        #     if(angle > 0 and angle < 45):
        #         left_side.append(self.calculate_centroid(c))
        #     elif(angle < 0 and angle > -45):
        #         right_side.append(self.calculate_centroid(c))

        # if(len(left_side) == 0 or len(right_side) == 0):
        #     return False, None, None

        # left_side = sorted(left_side, key=lambda c: c[0], reverse=False)
        # right_side = sorted(right_side, key=lambda c: c[0], reverse=True)

        # left = left_side[0]

        # # Target strip closest to 35% of the image width from the left
        # target_position = 0.35 * img_width

        # # Find left strip closest to targetp position
        # for l in left_side:
        #     if(abs(l[0] - target_position) < abs(left[0] - target_position)):
        #         left = l

        # right = right_side[0]

        # # Find right pair for the left strip
        # for r in right_side:
        #     if(r[0] < right[0] and r[0] > left[0]):
        #         right = r

        # if(right[0] > left[0]):
        #     return True, left, right
        # else:
        #     return False, None, None

    def calculate_offset(self, left, right, img_width):
        width = abs(right[0] - left[0])
        centroid = 0.5 * (left[0] + right[0])
        centroid += width * self.percent_offset * 0.5

        half_width = 0.5 * img_width

        offset = (centroid - half_width) / half_width

        return offset

    def process(self, inframe, outframe):
        imgbgr = inframe.getCvBGR()
        imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(imghsv, self.min_threshold, self.max_threshold)

        _, width, _ = imghsv.shape

        if not self.mask_only:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in contours:
                if(cv2.contourArea(c) > 8):
                    self.draw_contour(imghsv, c)

            detected, left, right = self.find_vision_targets(contours, width)

            if detected:
                offset = self.calculate_offset(left, right, width)

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

        _, width, _ = imghsv.shape

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)

        detected, left, right = self.find_vision_targets(contours, width)

        if detected:
            offset = self.calculate_offset(left, right, width)
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
