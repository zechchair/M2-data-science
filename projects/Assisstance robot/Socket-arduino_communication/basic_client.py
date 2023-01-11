import socket

def client():
    #host = socket.gethostbyname("ZECHCHAIR")  # get local machine name
    host = '10.1.32.113'  #you should use the same ip adress of the server here
    port = 8000                   # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    
    message = input('-> ')
    while message != 'q':
        s.send(message.encode('utf-8')) #send message 
        data = s.recv(1024).decode('utf-8')  #Gets the incomming message # 1023 is the length of the message 1023 letter of one message
        print('Received from server: ' + data)
        message = input('==> ')
    s.close() #close client connection


client()