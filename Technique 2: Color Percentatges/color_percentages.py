import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

# Update Method #1: calculate most common color groupings and filter by prevalence of that color within the frame
# I'd suggest uncommenting the matplotlib plot; it looks more impressive than a mask that doesn't work very well lol
video = cv2.VideoCapture('Inputs/few_plants.MOV')

while True:

    '''Reduce Image Size'''
    ret, img = video.read()
    if ret==False:
        break
    h,w,_ = img.shape
    min_dim = min(h,w)
    dim_ratio = max(h,w)/min_dim
    if min_dim == h:
        resh = (int(255*dim_ratio),255)
    else:
        resh = (255,int(255*dim_ratio))
    clean_frame = cv2.resize(img, resh, interpolation=cv2.INTER_AREA)

    # '''Draw Center Box: THE TARGET ZONE :o  '''

    # h,w,_ = frame.shape
    # if min(h,w) == h:
    #     target_pt1 = (int(0.5*w-0.25*h),int(0.25*h))
    #     target_pt2 = (int(0.5*w+0.25*h),int(0.75*h))
    # else:
    #     target_pt1 = (int(0.25*w),int(0.5*h-0.25*w))
    #     target_pt2 = (int(0.75*w),int(0.5*h+0.25*w))

    # cv2.rectangle(frame, target_pt1, target_pt2, (0,255,0), 1)

    img_hsv = cv2.cvtColor(clean_frame,cv2.COLOR_BGR2HSV)
    cv2.imwrite('Inputs/cig_frame_hsv.jpg', img_hsv)
    frame = Image.open('Inputs/cig_frame_hsv.jpg')
    pixels = list(frame.getdata())

    flat = [x for sets in pixels for x in sets]
    # r = flat[::4]
    # g = flat[1::4]
    # b = flat[2::4]
    a = flat[3::4]

    fig, ax = plt.subplots()
    # [0] is the # of instances, [1] is the color value
    # red = np.histogram(r, bins=255)
    # green = np.histogram(g, bins=255)
    # blue = np.histogram(b, bins=255)
    a_val = np.histogram(a, bins=255)

    '''Smooth it Out'''
    # red_vals = red[0]
    # green_vals = green[0]
    # blue_vals = blue[0]
    a_vals = a_val[0]
    for repeat in range(5):  
        # red_avg = []
        # green_avg = []
        # blue_avg = []
        a_avg = []
        for i in range(len(a_vals)-1):
            # red_avg.append(0.5*(red_vals[i]+red_vals[i+1]))
            # green_avg.append(0.5*(green_vals[i]+green_vals[i+1]))
            # blue_avg.append(0.5*(blue_vals[i]+blue_vals[i+1]))
            a_avg.append(0.5*(a_vals[i]+a_vals[i+1]))
        # red_vals = red_avg 
        # green_vals = green_avg
        # blue_vals = blue_avg
        a_vals = a_avg

    # ax.plot(red[0],color='red')
    # ax.plot(red_avg, color='red')
    # ax.plot(green[0],color='green')
    # ax.plot(green_avg, color='green')
    # ax.plot(blue[0],color='blue')
    # ax.plot(blue_avg, color='blue')
    ax.plot(a_avg, color='black')

    '''Peaks and Valleys'''
    pts_x = []
    pts_y = []
    for i in range(1,len(a_avg)-1):
        if (a_avg[i-1] < a_avg[i] and a_avg[i]>a_avg[i+1]) or (a_avg[i-1] > a_avg[i] and a_avg[i] < a_avg[i+1]):
            pts_x.append(i)
            pts_y.append(int(a_avg[i]))

    to_delete = []
    for i in range(1,len(pts_x)-1):
        try:
            dist1 = math.sqrt( ((pts_x[i]-pts_x[i-1])**2 + (pts_y[i]-pts_y[i-1])**2) )
            dist2 = math.sqrt( ((pts_x[i+1]-pts_x[i])**2 + (pts_y[i+1]-pts_y[i])**2) )
            if dist1<5 and dist2<5:
                    to_delete.append(pts_x[i])
        except IndexError:
            continue
    subt = 0
    for b in range(len(pts_x)):
        if pts_x[b-subt] in to_delete:
            del pts_x[b-subt]
            del pts_y[b-subt]
            subt+=1

    maxes_x, maxes_y, mins_x, mins_y = [], [], [], []
    mins_x.append(0)
    mins_y.append(a_avg[0])
    for i in range(1,len(pts_y)-1):
        if pts_y[i] >= pts_y[i-1] and pts_y[i] >= pts_y[i+1]:
            maxes_y.append(pts_y[i])
            maxes_x.append(pts_x[i])
        elif pts_y[i] < pts_y[i-1] and pts_y[i] < pts_y[i+1]:
            mins_y.append(pts_y[i])
            mins_x.append(pts_x[i])
    mins_x.append(250)
    mins_y.append(a_avg[-1])

    ax.scatter(maxes_x, maxes_y,color='red')
    ax.scatter(mins_x, mins_y,color='blue')

    zones = []
    zone_height=0
    for i in range(len(mins_x)-1):
        x1 = mins_x[i]
        x2 = mins_x[i+1]
        zone_width = x2 - x1
        for a in range(len(maxes_x)):
            if (maxes_x[a]>int(x1)) and (maxes_x[a]<int(x2)):
                zone_height = maxes_y[a]
        if zone_height!=0:
            zones.append(zone_width*zone_height)
        zone_height=0

    total_area = img.shape[0]*img.shape[1]

    found = []
    for i, zone in enumerate(zones):
        perc = zone/total_area*100
        if i==0 and perc<5:
            found.append([0,1])
        elif i>0 and perc<50 and perc>5:
            found.append([i,i+i])


    mask_lower = np.array([mins_x[found[0][0]], 0, 0],np.uint8)
    mask_upper = np.array([mins_x[found[0][1]], 255, 255],np.uint8)
    mask = cv2.inRange(img_hsv, mask_lower, mask_upper)
    product = cv2.bitwise_and(clean_frame, clean_frame, mask=mask)

    '''Find Shapes'''
    # gray_img = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray_img, (5,5),0)
    # ret, thresh = cv2.threshold(blur, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(clean_frame, contours, -1, (0,255,0), 1)

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour,True)

    cv2.imshow('Output',product)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

