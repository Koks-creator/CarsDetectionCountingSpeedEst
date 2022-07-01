from dataclasses import dataclass, field
from random import randint
from collections import deque
import math


@dataclass
class Tracker:
    max_distance: int = 32
    _track_id: int = field(init=False, default=0)
    _tracking_objects: dict = field(init=False, default_factory=dict)
    _used_colors: list = field(init=False, default_factory=list)

    def set_color(self):
        while True:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            if color not in self._used_colors:
                break

        self._used_colors.append(color)
        return color

    def update(self, center_points_cur_frame: list, center_points_prev_frame: list, frame_count: int):
        # tylko na pierwszym frame'ie
        if frame_count <= 2:
            for pt1 in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    # cv2.circle(img, pt1, 5, (0, 0, 255), -1)
                    # cv2.circle(img, pt2, 5, (0, 255, 255), -1)
                    distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

                    if distance < self.max_distance:
                        self._tracking_objects[self._track_id] = {"Center": pt1, "CenterHistory": deque(maxlen=20),
                                                                  "Color": self.set_color(), "StartTime": False,
                                                                  "EndTime": False}
                        self._track_id += 1
        else:
            tracking_objects_copy = self._tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id in tracking_objects_copy:
                object_exits = False
                pt2 = tracking_objects_copy[object_id]['Center']
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < self.max_distance:
                        self._tracking_objects[object_id]['Center'] = pt
                        self._tracking_objects[object_id]['CenterHistory'].append(pt)
                        object_exits = True

                        if pt in center_points_cur_frame:
                            # usuwanie punktow ktore sa by zostaly, tylko nowe
                            center_points_cur_frame.remove(pt)
                            continue

                # wywalamy id ktorych juz nie ma na ekranie
                if not object_exits:
                    self._tracking_objects.pop(object_id)

            # nowe id
            for pt in center_points_cur_frame:
                self._tracking_objects[self._track_id] = {"Center": pt, "CenterHistory": deque(maxlen=20),
                                                          "Color": self.set_color(), "StartTime": False,
                                                          "EndTime": False}
                self._track_id += 1

        return self._tracking_objects
