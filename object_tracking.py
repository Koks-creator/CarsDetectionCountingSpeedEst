from time import time
import cv2
import numpy as np

from object_detection2 import ObjectDetection
from tracker import Tracker
from kalmanfilter import KalmanFilter


kf = KalmanFilter()
od = ObjectDetection()
tracker = Tracker()
cap = cv2.VideoCapture(r"los_angeles.mp4")

# counting frames
count = 0

ptime = 0
center_points_prev_frame = []
p_detection = 0
c_detection = 0

end_area = [(229, 599), (995, 563), (955, 531), (257, 564)]
start_area = [(382, 396), (762, 389), (790, 410), (365, 420)]
start_area2 = [(975, 443), (1360, 410), (1290, 390), (926, 420)]
end_area2 = [(820, 359), (1110, 349), (1050, 332), (790, 343)]
areas_color = (0, 0, 200)
end_are_color = (0, 220, 0)

vehicles_counter = set()
too_fast_vehs = set()
distance1 = 20
distance2 = 20

max_speed = 100
while True:
    success, img = cap.read()
    count += 1
    if success is False:
        break

    img = cv2.resize(img, (1366, 768))
    overlay = img.copy()

    cv2.fillPoly(overlay, [np.array(start_area, np.int32)], color=(0, 0, 255))
    cv2.fillPoly(overlay, [np.array(end_area, np.int32)], color=(0, 0, 255))
    cv2.fillPoly(overlay, [np.array(end_area2, np.int32)], color=(0, 0, 255))
    cv2.fillPoly(overlay, [np.array(start_area2, np.int32)], color=(0, 0, 255))

    alpha = 0.4  # Transparency factor.

    # Following line overlays transparent rectangle
    # over the image
    final_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    # cv2.imshow("Res", xd)
    # Copy of points

    ctime = time()
    fps = int(1 / (ctime - ptime))
    ptime = ctime

    center_points_cur_frame = []
    boxes = od.detect(img, allowed_classes=[1, 2, 3, 5, 7])
    for box in boxes:
        x, y, w, h = box[:4]
        cx, cy = x + w // 2, y + h // 2
        p1 = x, y
        p2 = x + w, y + h

        center_points_cur_frame.append((cx, cy))

        cv2.circle(final_img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(final_img, p1, p2, (0, 255, 0), 1)
        line_w = min(int(w * 0.2), int(h * 0.2))

        cv2.line(final_img, (x, y), (x + line_w, y), (0, 0, 255), 4)
        cv2.line(final_img, (x, y), (x, y + line_w), (0, 0, 255), 4)

        cv2.line(final_img, (x + w, y), (x + w - line_w, y), (0, 0, 255), 4)
        cv2.line(final_img, (x + w, y), (x + w, y + line_w), (0, 0, 255), 4)

        cv2.line(final_img, (x, y + h), (x + line_w, y + h), (0, 0, 255), 4)
        cv2.line(final_img, (x, y + h), (x, y + h - line_w), (0, 0, 255), 4)

        cv2.line(final_img, (x + w, y + h), (x + w - line_w, y + h), (0, 0, 255), 4)
        cv2.line(final_img, (x + w, y + h), (x + w, y + h - line_w), (0, 0, 255), 4)

    tracking_objects = tracker.update(center_points_cur_frame, center_points_prev_frame, count)
    for object_id in tracking_objects:
        pt = tracking_objects[object_id]['Center']
        color = tracking_objects[object_id]['Color']

        cv2.circle(final_img, pt, 5, (0, 0, 255), -1)
        cv2.putText(final_img, str(object_id), (pt[0], pt[1] - 7), 0, .7, color, 2)

        # linia toru ruchu i predykcja kierunku
        points_history = tracking_objects[object_id]['CenterHistory']
        if len(points_history) > 2:
            for index, ppoint in enumerate(points_history):
                # Motion line
                if index != len(points_history) - 1:
                    cv2.line(final_img, points_history[index], points_history[index + 1], color, 4)

                # Predictions on previous points
                first_prediction = kf.Estimate(ppoint[0], ppoint[1])
                if index == len(points_history) - 1:

                    p_prediction = first_prediction

                    # Drawing extended prediction line
                    prediction_line_points = []
                    for i in range(5):
                        new_prediction = kf.Estimate(p_prediction[0], p_prediction[1])
                        if len(points_history) > 6:
                            cv2.line(final_img, p_prediction, new_prediction, (255, 255, 120), 3)

                        prediction_line_points.append(new_prediction)
                        p_prediction = new_prediction

                    if len(points_history) > 6:
                        # Strzalka
                        pointer_p1 = prediction_line_points[-2][0] - 8, prediction_line_points[-2][1]
                        pointer_p2 = prediction_line_points[-2][0] + 8, prediction_line_points[-2][1]
                        cv2.line(final_img, prediction_line_points[-1], pointer_p1, (255, 0, 255), 3)
                        cv2.line(final_img, prediction_line_points[-1], pointer_p2, (255, 0, 255), 3)

        # cv2.line(img, (378, 396), (762, 389), (0, 200, 0), 2)
        # cv2.line(img, (365, 420), (784, 410), (0, 200, 0), 2)
        # cv2.line(img, end_area[2], end_area[3], counter_line_color, 2)

        cv2.polylines(final_img, [np.array(start_area, np.int32)], True, areas_color, 2)
        cv2.polylines(final_img, [np.array(end_area, np.int32)], True, areas_color, 2)
        cv2.polylines(final_img, [np.array(start_area2, np.int32)], True, areas_color, 2)
        cv2.polylines(final_img, [np.array(end_area2, np.int32)], True, areas_color, 2)

        if len(points_history) > 3:
            result = cv2.pointPolygonTest(np.array(start_area, np.int32), pt, False)
            result2 = cv2.pointPolygonTest(np.array(start_area2, np.int32), pt, False)
            if result2 >= 0 or result >= 0:
                if tracking_objects[object_id]['StartTime'] is False:
                    cv2.circle(final_img, pt, 8, (255, 255, 255), -1)
                    tracking_objects[object_id]['StartTime'] = time()

        if len(points_history) > 3:
            result = cv2.pointPolygonTest(np.array(end_area, np.int32), pt, False)
            result2 = cv2.pointPolygonTest(np.array(end_area2, np.int32), pt, False)
            if result2 >= 0 or result >= 0:
                if tracking_objects[object_id]['EndTime'] is False:
                    cv2.circle(final_img, pt, 8, (255, 255, 255), -1)
                    tracking_objects[object_id]['EndTime'] = time()

                    if tracking_objects[object_id]['EndTime'] is not False and tracking_objects[object_id]['StartTime'] is not False:
                        elapsed_time = tracking_objects[object_id]['EndTime'] - tracking_objects[object_id]['StartTime']
                        if result >= 0:
                            a_speed_ms = distance1 / elapsed_time
                            a_speed_kh = a_speed_ms * 3.6 * int(20 / fps)
                            # print(object_id, a_speed_kh)

                            tracking_objects[object_id]['Speed'] = a_speed_kh
                        if result2 >= 0:
                            a_speed_ms = distance2 / elapsed_time
                            a_speed_kh = a_speed_ms * 3.6 * int(20 / fps)
                            # print(object_id, a_speed_kh)

                            tracking_objects[object_id]['Speed'] = a_speed_kh

                cv2.circle(final_img, pt, 8, (255, 255, 255), -1)
                vehicles_counter.add(object_id)

        if 'Speed' in tracking_objects[object_id].keys():
            speed = int(tracking_objects[object_id]['Speed'])
            if speed <= max_speed:
                cv2.putText(final_img, f"{speed}km/h", (pt[0], pt[1] - 30), 0, .7, color, 2)
            else:
                cv2.putText(final_img, f"{speed}km/h", (pt[0], pt[1] - 30), 0, .7, (0, 0, 200), 2)
                too_fast_vehs.add((object_id, speed))

        print(too_fast_vehs)

    cv2.putText(final_img, f"Max speed: {max_speed}km/h", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 100), 2)
    cv2.putText(final_img, f"Vehicles counter: {len(vehicles_counter)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 100), 2)
    cv2.putText(final_img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 100), 2)

    cv2.imshow("Res", final_img)
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
