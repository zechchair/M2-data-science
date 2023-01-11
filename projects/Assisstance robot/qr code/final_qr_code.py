import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import serial
import pyrealsense2 as rs
import time

#serial configuration
ser = serial.Serial('COM6', baudrate=9600)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
pipeline.start(config)


qr_data_pass = []
qr_code_base = 'base'
list_qr_code = ["b'Hello :)'", "b'01234567891011121314151617181920'", "b'Delta'", 'd', 'e', 'f']
list_qr_passed = ["b'Hello :)'", "b'01234567891011121314151617181920'", "b'Delta'", 'd', 'e', 'f']
teta = 1

def detect_qr(frame):
    list_qr = {}
    decodedObjects = pyzbar.decode(frame)
    if decodedObjects == []:
        return False
    for obj in decodedObjects:
        bbox = obj.rect
        x, y, w, h = bbox.left, bbox.top, bbox.width, bbox.height
        list_qr[(x, y, w, h)] = str(obj.data)
    return list_qr

def element_center(diff):
    return diff*0.109375

def passed(dico, dir_data):
    if dico == True:
        return True
    for dt in dir_data:
        if dt in dico.values() or (dt not in list_qr_code):
            return True
    return False

def get_dist(data):
    frame, frame_dpt = get_frames()
    etat = detect_qr(frame)
    for cle,valeur in etat.items():
        if valeur == data:
            x, y, w, h = cle
            break
    l_dist = []
    for i in range(x,x+w,10):
        for j in range(y,y+h,10):
            l_dist.append(frame_dpt.get_distance(i, j))
    return np.mean(l_dist)

def get_frames():
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    frame = frames.get_color_frame()
    frame = np.asanyarray(frame.get_data())
    frame_dpt = depth_frame.as_depth_frame()
    return frame, frame_dpt



def run_qr():
    teta = 1
    run = True
    while run:
        frame, frame_dpt = get_frames()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            run = False
        etat = detect_qr(frame)
        if not etat or passed(etat, qr_data_pass):
            if teta > 300:
                move = -20
            if teta < 60:
                move = 20
            teta = teta + move
            print(teta)
            zakaria= str(teta)+" \n "
            ser.write(zakaria.encode())
            time.sleep(1)
            continue
        
        for cle,valeur in etat.items():
            #qr code in target
            x, y, w, h = cle
            data = valeur

            if data in list_qr_passed:
                #center element
                center = element_center(int(320-(x+w/2)))
                teta = teta + center
                print(teta)
                zakaria= str(teta)+" \n "
                ser.write(zakaria.encode())
                time.sleep(3)

                #get distance
                distance = get_dist(data)
                print(distance)
                distance_char = str(distance+10000) + '/n'
                ser.write(distance_char.encode())
                time.sleep(1)

                #processing
                print(data)
                qr_data_pass.append(data)
                list_qr_passed.remove(data)
                if len(list_qr_passed) == 0:
                    list_qr_passed.append(qr_code_base)
                break
            else:
                if teta > 300:
                    move = -20
                if teta < 60:
                    move = 20
                teta = teta + move
                print(teta)
                zakaria= str(teta)+" \n "
                ser.write(zakaria.encode())
                time.sleep(1)
            
        print(len(list_qr_passed))
        
        if len(list_qr_passed) == 0:
            run = False  

run_qr()
