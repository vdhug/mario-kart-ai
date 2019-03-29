import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

# Countdown before the automatic action
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)



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
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    processed_img = roi(processed_img, [vertices])
    return processed_img


while True:
    # Screen 640 x 480
    screen =  np.array(ImageGrab.grab(bbox=(0,40,640,480)))
    new_screen = process_image(screen)

    # Output image of AI
    # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    cv2.imshow('new window', new_screen)
    print('Loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
