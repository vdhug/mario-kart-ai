import numpy as np
from PIL import ImageGrab
import cv2
import time

last_time = time.time()

while True:
    # Screen 640 x 480
    screen =  np.array(ImageGrab.grab(bbox=(0,40,640,480)))

    # Output image of AI
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    print('Loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
