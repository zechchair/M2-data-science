
import socket
import time

''' make sure u unifie the ip adress
you should run server first

it will be better if the timeout is bigger
'''


def server_read(Port,timeout):
        
    IP = '10.1.32.113'     # this ip is the server machine ip you can get it using cmd and execute ipconfig ( if the server and the client are both on the same machine you can also use the local ip adress 127.0.0.1)
    #IP = socket.gethostname() #IP address of local machine   '''it  returns 127.0.0.1 or your machine name" 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Creates an instance of socket
    #you should use the same port in your client program client and you can choose any number to be your port
    maxConnections = 999

    s.setblocking(False) #we use it before a command to specify if the program should wait for this command to be true or it should just try it then it passes
    s.bind(('',Port))
    connected = False
    s.settimeout(timeout) #the timeout of looking for a new client , if the program couldn't find a new client within this time it will start looking for client with a new iteration
    s.listen(maxConnections)
    print("s started at " + IP + " on port " + str(Port))
    while not connected:

            try:
                s.settimeout(timeout) #the timeout of looking for a new client , if the program couldn't find a new client within this time it will start looking for client with a new iteration
                (clientsocket, address) = s.accept() #adress is the adress of client
                #print("New connection made!")
                s.settimeout(timeout) #the timeout of looking for a new client , if the program couldn't find a new client within this time it will start looking for client with a new iteration
                message = clientsocket.recv(1023).decode() #Gets the incomming message
                if len(message)!=0:
                    connected = True
            except:
                pass
    return message
    s.close()


def server_write(Port,message,timeout):
        
    IP = '10.1.32.113'     # this ip is the server machine ip you can get it using cmd and execute ipconfig ( if the server and the client are both on the same machine you can also use the local ip adress 127.0.0.1)
    #IP = socket.gethostname() #IP address of local machine   '''it  returns 127.0.0.1 or your machine name" 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Creates an instance of socket
    #you should use the same port in your client program client and you can choose any number to be your port
    maxConnections = 999

    s.setblocking(False) #we use it before a command to specify if the program should wait for this command to be true or it should just try it then it passes
    s.settimeout(timeout) #the timeout of looking for a new client , if the program couldn't find a new client within this time it will start looking for client with a new iteration
    s.bind(('',Port))
    connected = False
    s.listen(maxConnections)
    print("s started at " + IP + " on port " + str(Port))
    while not connected:
        
            try:
                s.settimeout(timeout) #the timeout of looking for a new client , if the program couldn't find a new client within this time it will start looking for client with a new iteration
                (clientsocket, address) = s.accept() #adress is the adress of client
                #print("New connection made!")
                clientsocket.send(message.encode())
                connected = True
            except:
                pass

    s.close()

print(server_read(8000,100))
server_write(8000,"hi zakaria",100) 
