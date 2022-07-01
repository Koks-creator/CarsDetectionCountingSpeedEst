from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class ObjectDetection:
    weights_path: str = "dnn_model/yolov4-tiny.weights"
    cfg_path: str = "dnn_model/yolov4-tiny.cfg"
    classes_file = r"dnn_model/classes.txt"
    nms_threshold: int = .3
    conf_threshold: int = .3
    image_w: int = 736 #wielokrotnosc 32
    image_h: int = 736

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            class_names = f.read().rstrip('\n').split("\n")
            return class_names

    def detect(self, img: np.array, allowed_classes=False, draw=False):
        classes_list = self.load_classes()

        ih, iw, _ = img.shape
        bbox = []
        class_ids = []
        confs = []

        if allowed_classes is False:
            allowed_classes = [i for i in range(len(classes_list))]

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.image_w, self.image_h), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_names)

        for output in outputs:
            for det in output:
                # print(cos[5:])
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id in allowed_classes:
                    if confidence > self.conf_threshold:
                        w, h = int(det[2] * iw), int(det[3] * ih)
                        x, y = int((det[0] * iw) - w / 2), int((det[1] * ih) - h / 2)

                        bbox.append([x, y, w, h])
                        class_ids.append(class_id)
                        # print(confidence)
                        confs.append(float(confidence))
                        # print(confs)

        # to wywala zbyt duza ilosc bboxow, zwraca id odpowiedniego bboxa w liscie bboxow
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.conf_threshold, self.nms_threshold)

        bbox_list = []
        for i in indices:
            i = i[0]  # bo i jest lista, np [1]
            # print(confs[i])
            box = bbox[i]
            # print(class_ids)
            # print(classes_list)
            x, y, w, h = box[0], box[1], box[2], box[3]
            class_name = classes_list[class_ids[i]].upper()
            bbox_list.append([x, y, w, h, class_name])
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (240, 100, 255), 2)
                cv2.putText(img, f"{classes_list[class_ids[i]].upper()} {int(confs[i] * 100)}%", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return bbox_list


if __name__ == '__main__':
    ob = ObjectDetection()
    print(ob.load_classes())
