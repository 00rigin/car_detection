# Car detection using YOLO #
This is custom Darknet.
It can use by C or python.

<python>
If you want to run with python, go to ./python/darknet.py
edit video path "cv2.VideoCapture('path')"
edit net
edit meta
after save, cd into python. and
$:python darknet.py 

<C>
If you want to run with C,
edit at Makefile, OPENCV=1
after save 
$:make
$:./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights /"video path"


Here is the list what I change :

1. print detected object's center axis

2. Input video are cuted. (1/2 vertical, 1/3 horizon).
at <c> You can change the size at 'get_image_from_stream()' in the 'src/image_opencv.cpp'
at <python> You can change the size at main code in the 'python/darknet.py'

3. We can capture frame and inside of bounding Box.
at <C> unfortunately if object are detected, snap shot will saved at 'snap_detedted'. I didnt amend yet. if you want to fix it, go into './src/image_opencv.cpp' and fix 'snap_shot()'
at <python> you can capture frame and inside of bounding box.
when object are deteted, press 'c' than capture will saved at ./snap_detected

4. This code print the center axis of deteced object (only car, bus, truck).
you can change at './src/image_opencv.cpp' line 384.

5. only objects of a certain size or more are marked.
at <c> you can change at './src/image.c' 'draw_detection' line 369
at <python> you can change at './python/darknet.py', check 'def draw()'
 








