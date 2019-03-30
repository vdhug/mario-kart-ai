import numpy as np
import cv2
import time
import os
from grabscreen import grab_screen
from getkeys import key_check


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


last_time = time.time()

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_image(original_image):
    # convert to gray
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    vertices = np.array([[10,300],[10,200],[300,100],[500,100],[800,200],[800,500],
                         ], np.int32)
    processed_img = cv2.GaussianBlur(processed_img,(5,5),0)
    processed_img = roi(processed_img, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                          edges       rho   theta   thresh         # min length, max gap:
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]),      150,         5)
    return processed_img


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


# Countdown before the automatic action
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
while True:
    # Screen 640 x 480
    screen = grab_screen(region=(0,40,640,480))
    # print('Loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()

    # Convert screen to gray scale
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (80,60))
    keys = key_check()
    output = keys_to_output(keys)
    training_data.append([screen,output])

    if len(training_data) % 500 == 0:
        print(len(training_data))
        np.save(file_name, training_data)
