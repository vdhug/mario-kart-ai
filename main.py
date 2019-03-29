import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time = time.time()


def process_image(original_image):
    # convert to gray
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
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
