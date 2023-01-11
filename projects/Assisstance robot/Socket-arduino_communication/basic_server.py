import socket
import time
''' 
this is a basic example of the socket connection
you can send a message here and wait for the answer and unfortunatly you cannot send two messages at once the answer is required

'''


host = '10.1.33.40'    # this ip is the server machine ip you can get it using cmd and execute ipconfig ( if the server and the client are both on the same machine you can also use the local ip adress 127.0.0.1)
#IP = socket.gethostname() #IP address of local machine   '''it  returns 127.0.0.1 or your machine name"
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Creates an instance of socket
Port = 8000 #Port to host s on
maxConnections = 999
IP = host #IP address of local machine
s.settimeout(100)
s.bind(('',Port))
connected = False
print("Server not connected")
s.listen(maxConnections) #open connection
print("s started at " + IP + " on port " + str(Port))
(clientsocket, address) = s.accept()
while True:
    try:
        message = clientsocket.recv(1023).decode('utf-8') #Gets the incomming message # 1023 is the length of the message 1023 letter of one message
        print("received from client : " +message)
        to_send = input("say what you want to send from server: ") #Gets the message to be sent
        clientsocket.send(to_send.encode('utf-8')) #send the message

    except:
      pass
s.close() #to close this connection