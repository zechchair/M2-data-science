import socket
import time

''' this program helps to receive data from client (using client_sending_with_connection_interruption program in this folder) 
it helps if you want to receive data and you risk that connection network could be interrupted 
so client connection will be closed after every sent of data and server program will look again for the client after it receives the data

you can play with those algorithms to customize your socket connection

for more info please check this website https://docs.python.org/3/library/socket.html
'''



IP = '10.1.32.113'     # this ip is the server machine ip you can get it using cmd and execute ipconfig ( if the server and the client are both on the same machine you can also use the local ip adress 127.0.0.1)
#IP = socket.gethostname() #IP address of local machine   '''it  returns 127.0.0.1 or your machine name" 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Creates an instance of socket
Port = 8000 #you should use the same port in your client program client and you can choose any number to be your port
maxConnections = 999

s.setblocking(False) #we use it before a command to specify if the program should wait for this command to be true or it should just try it then it passes
s.settimeout(100) #the timeout of looking for a new client , if the program couldn't find a new client within this time it will start looking for client with a new iteration
s.bind(('',Port))
connected = False
print("Server not connected")
s.listen(maxConnections)
print("s started at " + IP + " on port " + str(Port))
while True:
  if(not connected):
    try:
        (clientsocket, address) = s.accept() #adress is the adress of client

        print("New connection made!")
        connected = True
    except:
        pass
  else:
    try:
        message = clientsocket.recv(1023).decode() #Gets the incomming message
        if len(message)!=0:
              print(message)
        else:
              connected=False
        connected = False #if you want to preserve the connection please don't comment this line (and after every communication the derver will be deconected and it will try to connect again with the client)
    except:
        print("client not connected")
        connected = False
        pass
s.close()