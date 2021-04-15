import cv2 as cv
import numpy as np
import time

# Initialize the parameters
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4  # Non-maximum suppression threshold
inp_width = 288  # Width of network's input image
inp_height = 288  # Height of network's input image
class_file = "model_hand.names"
config = "tiny3l-5.cfg"
weights = "Best.weights"
with open(class_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
net = cv.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)      #Uncomment to use GPU after installed cuda dependecies
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)        #Uncomment to use GPU after installed cuda dependecies


# this method returns the layers of the YOLO network that is responsible for the object detection
def get_output_layers(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# this method is responsible for drawing the bounding boxes
def draw_pred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# this method is responsible for gathering information about each detection, and sending the information to the
# draw_prediction method
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                print(left, top)
                mouse_control(classId, center_x, center_y, width, height)
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_pred(classIds[i], confidences[i], left, top, left + width, top + height, frame)


# Mouse movements and controls go in this method once implemented
def mouse_control(id, x, y, width, length):
    print(x, y, width, length)


def main():
    win_name = 'Deep learning object detection in OpenCV'
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cap = cv.VideoCapture(0)
    time_now = time.time()
    frameid = 0
    while True:
        hasFrame, frame = cap.read()
        frameid += 1
        if not hasFrame:
            cap.release()
            break
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inp_width, inp_height), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        postprocess(frame, outs)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        end_time = time.time() - time_now
        fps = frameid / end_time
        cv.putText(frame, "FPS: " + str(fps), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv.imshow(win_name, frame)
        k = cv.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
