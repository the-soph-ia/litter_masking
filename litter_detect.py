import cv2
import numpy as np

# specify the file that's being read
video = cv2.VideoCapture('patchy_ground.MOV')
# 'few_plants.MOV'
# 'cigarette.MOV'
# 'patchy_ground.MOV'

# loop through each frame
while True:
    ret, img_frame = video.read()
    if ret==False:
        break
    hsv_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
    hsv_frame[:,:,2] -= 60

# RED
    red_lower = np.array([160,100,150],dtype=np.uint8)
    red_upper = np.array([180,255,255],dtype=np.uint8)
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

# ORANGE/YELLOW
    orange_lower = np.array([5, 50, 50],np.uint8)
    orange_upper = np.array([30, 255, 255],np.uint8)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

# GREEN
    green_lower = np.array([30,80,100],dtype=np.uint8)
    green_upper = np.array([90,255,255],dtype=np.uint8)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

# BLUE
    blue_lower = np.array([105,153,160],dtype=np.uint8)
    blue_upper = np.array([140,255,255],dtype=np.uint8)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

# WHITE
    white_lower = np.array([0, 0, 190],np.uint8)
    white_upper = np.array([359, 50, 255],np.uint8)
    white_mask = cv2.inRange(hsv_frame, white_lower, white_upper)

    white = cv2.bitwise_and(img_frame, img_frame, mask=white_mask)
    blue = cv2.bitwise_and(img_frame,img_frame,mask=blue_mask)
    both = cv2.bitwise_or(white,blue)

# Filter out the white and blue shades from the image
    cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    img_resized = cv2.resize(img_frame, (int(img_frame.shape[1]*0.5),int(img_frame.shape[0]*0.5)))
    output_final = cv2.resize(white, (int(white.shape[1]*0.5),int(white.shape[0]*0.5)))
    cv2.imshow("output", np.hstack([img_resized, output_final]))
    # Close the window if the Q key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

