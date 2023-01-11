import cv2 as cv

#capture = cv.VideoCapture(0)
import urllib.request as ur
#capture = cv.VideoCapture("http://desktop-id0n14p:8080/cam_1.cgi")
#for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 

capture = cv.VideoCapture(0) #id 0 when you use your laptop camera but when you connect realsense camera your laptop camera id will be 1 and realsense id will be 2 ("without using pyrealsense library")

while True:
    isTrue,frame = capture.read() #isTrue become true if the camera is connected otherwise it will be false , and your frame is obviosly in frame variable
    cv.imshow('Video',frame) #to show your frames using opencv (open a window)
    if cv.waitKey(20) & 0xFF==ord('d'): #if you press on 'd' yoou will quit the window
        break

capture.release()
cv.destroyAllWindows()