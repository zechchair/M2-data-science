import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import serial
import pyrealsense2 as rs
import time

#serial configuration
ser = serial.Serial('COM6', baudrate=9600)
"""while True:
    x = str(input('donner x: '))
    print(x)
    ser.write(x.encode())"""


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
# Start streaming
pipeline.start(config)

qr_data_pass = []
teta = 1
font = cv2.FONT_HERSHEY_PLAIN

def detect_qr(frame):
    list_qr = {}
    decodedObjects = pyzbar.decode(frame)
    if decodedObjects == []:
        return False
    for obj in decodedObjects:
        bbox = obj.rect
        
        x, y, w, h = bbox.left, bbox.top, bbox.width, bbox.height
        #distance_to_hand = frame_dpt.get_distance(x, y)
        list_qr[(x, y, w, h)] = str(obj.data)
        cv2.rectangle(frame, (x,y),(x+w, y+h),(255, 0, 0), 8)
        cv2.putText(frame, str(obj.data), (50, 50), font, 2,
                    (255, 0, 0), 3)
    return list_qr

def element_center(diff):
    return diff*0.109375

def passed(dico, dir_data):
    if dico == True:
        return True
    for dt in dir_data:
        if dt in dico.values():
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

"""
cap = cv2.VideoCapture(0)

_, frame = cap.read()
H, W, _ = frame.shape"""
center_w = 320
print(center_w)

while True:
    
    # Wait for a coherent pair of frames: depth and color
    #frames = pipeline.wait_for_frames()
    #depth_frame = frames.get_depth_frame()
    #frame = frames.get_color_frame()
    #frame = np.asanyarray(frame.get_data())
    #frame_dpt = depth_frame.as_depth_frame()
    frame, frame_dpt = get_frames()
    #_, frame = cap.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
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
        x, y, w, h = cle
        data = valeur
        if data in qr_data_pass:
            continue
    
        center = element_center(int(320-(x+w/2)))
        time.sleep(4)
        teta = teta + center
        distance = get_dist(data)
        distance_char = str(distance) + '/n'
        print(teta)
        print(distance)
        ser.write(distance_char.encode())
        time.sleep(1)
        ser.write(str(distance+10000).encode())
        print(cle)
        qr_data_pass.append(data)
        break
    
    print("dezt")
    


    
