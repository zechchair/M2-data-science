import serial,time
ser = serial.Serial('COM17', baudrate = 9600, timeout = 1) # baudrate is a parametre of serial.begin in arduino, timeout is the time to try connction with arduino if the port isn't open it returns timeout error
arduinoData=""
while True:

    to_send=str("her what you want to send")+"\n"
    ser.write(to_send.encode('ascii'))  # this one to send data to arduino you can delete ascii if it disturbs the program
    arduinoData = ser.readline().decode('ascii') #read data sent from arduino
    time.sleep(0.1)#this one is optional just to make sure that connection is good
    if arduinoData!="":
        print(int(arduinoData))  #if arduino sent something it will be not empty


"""
in order to send something to arduino, try to send it and wait for returns for example 

send  ("toarduino" + data)  
and wait for (received) -- which should be sent by arduino whenever it received your "toarduino" as a string

and in the arduino part you should program it to read the first string and if it's equal to "toarduino" it will save data into a variable and send to python  "received"

#######################


and the same if you want to receive something from arduino for instance

send ("requireddata")
and wait for your data to come as a string


and in arduino part you should program it to test if it received "requireddata" , if so, it will send your required data to python, and you should save it in python

"""

'''
indeed we can name this algorithm a true communication, far from buggs
the most important thing in communication between multiple programmes is assuring the sent and the reception of the information , and the only way to do this is by establish a true communication instead of just send the message and hoping that it will be received
cuz in small programmes it works but when we put the same script in a big program we can't even know where is the problem
be smart and program it without hopes but with assurance

and this is an example of sending data to arduino

'''

def arduino_send(data):
    while True:
        ser.write(data.encode('ascii'))
        if ser.readline().decode('ascii') =="received":
            break


        
        