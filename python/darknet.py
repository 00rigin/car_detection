#-*- conding: utf-8 -*-
from ctypes import *
import math
import cv2
import os
import socket
import sys
import time
import numpy as np
import random
from datetime import datetime
import lane_detect # 만든것


def connect_srv(time, path1, path2):
    host = "192.168.0.143" 
    port = "9999"

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print('connect success')

    send_file(time, path1, path2)
    client_socket.close()



def send_file(time, path1, path2):
    

    file_1 = open(path1, "rb")
    file_2 = open(path2, "rb")
    img_size_1 = os.path.getsize(path1)
    img_size_2 = os.path.getsize(path2)
    img_1 = file_1.read(img_size_1)
    img_2 = file_2.read(img_size_2)
    file_1.close()
    file_2.close()

    client_socket.sendall(time)
    client_socket.sendall(img_1)
    client_socket.sendall(img_2)

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
    
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

# 19.11.10 
ndarray_image = lib.ndarray_to_image 
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)] 
ndarray_image.restype = IMAGE
#snap_shot = lib.snap_shot

# # 19.11.10 
def nparray_to_image(img): 
    data = img.ctypes.data_as(POINTER(c_ubyte)) 
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides) 
    
    return image


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    # 19.11.10 
    im = nparray_to_image(image)

    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

# # 19.11.10 
def convert_box_value(r): 
    boxes = [] 
    
    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 
        
        boxes.append((x, y, w, h)) 
    
    return boxes


# 19.11.11
# 시그널 받아서 처리 할 부분
def gen_flag(x,y, vari):
    #if (cv2.waitKey(33) == ord('c')):
    #    return 1
    #else: return 0

    l_lines = lane_detect.send_to_darknet()
    line_x = lane_detect.compute_model_inverse(l_lines, y)
    if x<float(line_x+vari) and x>float(line_x):
        return 1
    else: return 0




# 19.11.11
def sanp_shot(flag, image, top, bottom, left, right, number):
    snap_path = '../snap_detected/snap'
    roi_snap_path = '../snap_detected/roi_snap'

    #시간 string으로 넘겨 줄때 사용
    now = datetime.now()
    time = now.year+now.month+now.day+now.hour+now.minute+now.second
    

    if(flag == 1):
        number+=1
        snap_path = snap_path + str(number) + '.jpg'
        roi_snap_path = roi_snap_path + str(number) + '.jpg'
        cv2. imwrite(snap_path ,  image)
        snap = image[ top : bottom , left : right ]
        cv2. imwrite(roi_snap_path ,  snap)
    #    connect_srv(time, snap_path, roi_path)

    
    return number
        
# # 19.11.10 
def draw(image, boxes, number): 
    for k in range(len(boxes)): 
        x, y, w, h = boxes[k] 
        left = max(0, np.floor(x + 0.5).astype(int)) 
        top = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        center_x = left + int(w / 2)
        center_y = top + int(h*3/4)
        flag = 0
        height = image.shape[0]
        width = image.shape[1]
        # change object size
        if( w >= width*0.1 and h >= height*0.1389):
            if( ( center_y > 0 and center_y < height ) and ( center_x > 0 and center_x < width ) ):
                #print(center_x, center_y )
                # lane_detect 에서 line 받기.
                # lane_detect 로 보내서 확인하기.
                # gen_flag 가 차선 침범 확인함.
                flag = gen_flag(center_x, center_y, 10)
                if flag:
                    number = sanp_shot(flag, image, top, bottom, left, right, number)
                cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2) 
                cv2.circle(image, (center_x, center_y), 5, (0,0,255), 5)

    return number
    

# 19.11.10 
if __name__ == "__main__": 

   
    net = load_net(b"../cfg/yolov2-tiny.cfg", b"../yolov2-tiny.weights", 0) 
    meta = load_meta(b"../cfg/coco.data") 
    
    #cap = cv2.VideoCapture('../../blackbox_data/sample4.mkv')
    cap = cv2.VideoCapture('../../blackbox_data/test3.mov')
    
    ret, frame = cap.read()
    height, width, channel = frame.shape   
    global number
    number = 0


    """
    ret, frame = cap.read()
    if frame.shape[0] !=540: # resizing for challenge video
        frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA) 
    result = detect_lanes_img(frame)

    cv2.imshow('result',result)

    #out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    """

    while(cap.isOpened()): 

        ret, frame = cap.read() 
        # change video size
        #frame = frame[ int(height*0.5):int(height),  int(width*0.2) : int(width*0.8)]
        
        if not ret: 
            break 
        #print(type(frame)) 

        # lane_detect 시작
        if frame.shape[0] !=540: # resizing for challenge video
            frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA) 
        result = lane_detect.detect_lanes_img(frame)
        
        # 차량 검출 시작
        r = detect(net, meta, frame)


        boxes = convert_box_value(r) 

        number = draw(frame, boxes, number) 

        _frame = cv2.addWeighted(frame, 0.8, result, 1, 0.)
        cv2.imshow('frame', _frame) 
        
        if (cv2.waitKey(33) == ord('q')):
            break


"""

추가할 사항 : 
영상 크기 화면 비율에 맞추기............................0
영상 입력 받을때 좌우 위 아래 잘라내기...................0
sanp_shot 함수 구현하기................................0
roi_sanp 구현하기......................................0
일정 객체 크기 이상일때만 나오게 구현하기................0

"""
