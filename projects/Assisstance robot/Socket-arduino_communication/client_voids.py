import socket
import time

''' make sure u unifie the ip adress
you should run server first
it will be better here if the timeout is small

timeout is time for program to try to connect with server

'''


def client_read(port,timeout): # read what is sent for the client by server in this port with this timeout and the fuction returns the message after making sure that the message is well received
    host = '10.1.32.113'                   # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    message=""
    #print("Server not connected")
    while not connected:
        try:
            s.settimeout(timeout)
            s.connect((host, port))
            #print("Server connected")
            s.settimeout(timeout)
            message = s.recv(1023).decode()
            #print("client read done")
            s.close()
            if len(message)!=0 and message!="":
                #print("the message is : " +message)
                connected = True
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
        except:
           # print("Server not connected")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connected = False
            pass
    return message


def client_write(port,message,timeout):#write from client into server with making sure that the message is well received
    host = '10.1.32.113'                   # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    while not connected:
        try:
            s.settimeout(timeout)
            s.connect((host, port))
            s.send(message.encode()) #Encodes and sends message (x)
            #print("client write done")
            s.close()
            connected = True
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
        except:
            #("Server not connected")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connected = False
            pass
client_write(8000,"hello",0.1)
print(client_read(8000,0.1))
