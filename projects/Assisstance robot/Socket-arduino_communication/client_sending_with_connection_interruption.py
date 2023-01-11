import socket
import time
'''   
read the description in server_receiving_with_connection_interruption "in the same folder"

'''

host = '10.1.32.113'  #you should use the same ip adress of the server here
port = 8000                   # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connected = False
print("Server not connected")
while True:

    x = input("say what you want to send from client: ") #Gets the message to be sent

    try:
        s.settimeout(0.01) #the client will try to connect to server for this time
        s.connect((host, port)) #after the timeout and if the client doesn't find a server it will returns timeout and thanks to "try" it will try again to be connected
        print("Server connected")
        s.send(x.encode()) #Encodes and sends message (x)
        s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        
    except:
        print("Server not connected")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        connected = False
        pass

s.close()
