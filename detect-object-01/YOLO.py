import time
import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
args = ap.parse_args()


CLASSES_PATH = './yolo/yolov3.txt'
WEIGHTS_PATH = './yolo/yolov3.weights'
CONFIG_PATH = './yolo/yolov3.cfg'


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print('--layer_names--')
    print(len(output_layers), output_layers)

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# đọc ảnh
image = cv2.imread(args.image)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# đọc các đối tượng cần detect
classes = None
with open(CLASSES_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# Random màu ngẫu nhiên cho các class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the YOLO network model
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

# Create a blob
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# determine only the *output* layer names that we need from YOLO
# Xác định các lớp cần từ YOLO và nạp vào network
outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# Thực hiện xác định bằng HOG và SVM
start = time.time()
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

            print('--scores--')
            print('scores', scores, 'class_id', class_id)
            print('center_x', center_x, 'center_y', center_y)
            print('x', x, 'y', y, 'w', w, 'h', h)
            print('class_ids', class_ids)
            print('confidences', confidences)
            print('boxes', boxes)

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print('--indices--')
print(indices)

for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

end = time.time()
print("YOLO Execution time: " + str(end-start))

cv2.waitKey()
cv2.imwrite("./output/object-detection.jpg", image)
cv2.destroyAllWindows()
