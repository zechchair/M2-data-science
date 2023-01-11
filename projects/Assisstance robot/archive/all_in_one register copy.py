import smtplib
import wolframalpha
from textblob import TextBlob
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import math
from queue import PriorityQueue
import time
import argparse
import cv2
import numpy as np
from yolo import YOLO
import serial
ser = serial.Serial('COM17', baudrate = 9600, timeout = 1)
import wave
import contextlib
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import speech_recognition as sr
import pyttsx3
import nltk
import warnings
import datetime
import webbrowser
import time
from time import ctime
import os
from os import path
import socket
import wikipedia
import subprocess
import wolframalpha
import json
import requests

import time
import pyrealsense2 as rs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
r = sr.Recognizer()
browserExe = "chrome.exe"
list_max = []
Noms = []
s=0 #the number that helps to know when excuting the voice function what i should excute the main boucle or a boucle inside it
start_time=0
n=0
a=""

client =wolframalpha.Client('QXGQJX-K97LHHGQ3L')
hand=0
vd = cv2.VideoCapture(0)
#isTrue, frame=vd.read()



# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)




ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
yolo.size = int(args.size)
yolo.confidence = float(args.confidence)



#load mask model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

def wolf(query):

    query1=TextBlob(query).translate(to='en')
    #print(query1)
    res=client.query(query1)
    output=next(res.results).text
    #print(output)
    output2=TextBlob(output).translate(to='fr')
    print(output2)
    francais.speak(str(output2))

def mail_send(mail_content):
    #The mail addresses and password
    sender_address = 'eminia.robot@gmail.com'
    sender_pass = 'EMINIA2020'
    receiver_address = 'ayoub.talibi@emines.um6p.ma'
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'A test mail sent by Python. It has an attachment.'   #The subject line
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')

def battery_level():

    while True:
        b=str(1234)+"\n"
        ser.write(b.encode())
        data = ser.readline().decode('ascii')
        if data!="":
            battery=int(data)
            break
    return battery



def framerealsense():
    frame = pipeline.wait_for_frames()
    frame = frame.get_color_frame()
    frame = np.asanyarray(frame.get_data())
    return frame


def handisdetected(frame):
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
       return True

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()


    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.8:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) == 1:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def run():
    global clientsocket
    mask=False
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        isTrue, frame=vd.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            if (withoutMask > 0.80):
                label = "No Mask"
                color = (0, 0, 255)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                print("withoutMask")
                mask=False
                clientsocket.send("nocart,maskfalse,".encode())
            elif (mask > 0.90):
                print( "mask")
                mask=True
            
        if mask==True:
            print("here")
            mask=False
            break


def server(Port):
    global clientsocket,server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Creates an instance of socket
    maxections = 999
    IP ='127.0.0.1'   #IP address of local machine
    server.settimeout(100)
    server.bind(('',Port))
    ected = False
    print("ect processing")

    while not ected:
        try:
            server.listen(maxections)
            print("s started at " + IP + " on port " + str(Port))
            (clientsocket, address) = server.accept()
            print("server New ection made!")
            ected = True
        except:
            pass

def socket_send(text):
    global clientsocket
    to_send=str(arduinoData)+","+text
    while True:
        clientsocket.send(to_send.encode())
        if clientsocket.recv(1024).decode()=="rcvd":
            break

def register():
        print('register')
        freq = 41000
        #i should calculate the duration using the volume
        duration = 6
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        write("C:\\test1.wav", freq, recording)
        fname = 'C:\\test1.wav'
        outname = 'C:\\filtered1.wav'
        cutOffFrequency = 200.0
        def running_mean(x, windowSize):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
        def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
            if sample_width == 1:
                dtype = np.uint8  # unsigned char
            elif sample_width == 2:
                dtype = np.int16  # signed 2-byte short
            else:
                raise ValueError("Only supports 8 and 16 bit audio formats.")
            channels = np.frombuffer(raw_bytes, dtype=dtype)
            if interleaved:
                # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
                channels.shape = (n_frames, n_channels)
                channels = channels.T
            else:
                # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
                channels.shape = (n_channels, n_frames)
            return channels
        data, samplerate = sf.read(fname)
        sf.write(fname, data, samplerate, subtype='PCM_16')
        with contextlib.closing(wave.open(fname, 'rb')) as spf:
            sampleRate = spf.getframerate()
            ampWidth = spf.getsampwidth()
            nChannels = spf.getnchannels()
            nFrames = spf.getnframes()
            # Extract Raw Audio from multi-channel Wav File
            signal = spf.readframes(nFrames * nChannels)
            spf.close()
            channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)
            # get window size
            freqRatio = (cutOffFrequency / sampleRate)
            N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)
            # Use moviung average (only on first channel)
            filtered = running_mean(channels[0], N).astype(channels.dtype)
            wav_file = wave.open(outname, "w")
            wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
            wav_file.writeframes(filtered.tobytes('C'))
            wav_file.close()



class hind_class:
    def boucle():
        #text1 is the one we get from the voice function
        words1 = text1.split() #the sentence to a list of words
        converted_words1 = [x.upper() for x in words1] #the words' list en majuscule
        words = converted_words1 #new list of words with all the words majuscule
        print(words)
        exit_list = ['REVOIR', 'BYE', 'PROCHAINE']
        text2 = set(words) & set(exit_list) #setting the intersection between the two lists words and exit_list
        str_val = " ".join(text2) #making the intersection words a string to use it
        text02 = str_val #the new text variable we going to work with
        #two conditions whether the visitor wanna leave or complete
        if text02 in exit_list:
            francais.speak("au revoir, et bienvenue à emines")
        else:
            #define the list we using to match the departement
            list1 = ['RÉCEPTION', 'ENTRÉE', 'BLOC', 'A']
            list2 = ['NICOLAS', 'CHEIMANOFF', 'KHADIJA', 'AITHADOUCH', 'FRÉDÉRIC', 'DIRECTION',
                    'DIRECTION EMINES', 'NICO', 'SAAD', 'KHATAB', 'B']
            list3 = [ 'FATIHA', 'ABDELAOUI', 'C', 'FOYER', 'SCOLARITÉ','ZINEB']
            list4 = [ 'BLOC', 'D', 'REDA', 'BOUCHIKHI', 'HAJAR', 'KHOUKH']
            list5 = ['POLE', ' SANTÉ', 'BLOC', 'E', 'MÉDECIN', 'SANITAIRE', 'INFIRMERIE']
            list6 = ['ETECH', 'LABO', 'LABORATOIRE', 'BLOC']
            a = []
            hind = ['Réception', 'département direction', 'département de scolarité',
                    'département logistique', 'département santé', 'E-tech'] #list of the departement names
            #setting the matching words between the original words list we got from the visitor text said and the lists of departements
            list7 = sorted(set(list1) & set(words), key=lambda k: list1.index(k))
            a.append(list7)
            list8 = sorted(set(list2) & set(words), key=lambda k: list2.index(k))
            a.append(list8)
            list9 = sorted(set(list3) & set(words), key=lambda k: list3.index(k))
            a.append(list9)
            list10 = sorted(set(list4) & set(words), key=lambda k: list4.index(k))
            a.append(list10)
            list11 = sorted(set(list5) & set(words), key=lambda k: list5.index(k))
            a.append(list11)
            list12 = sorted(set(list6) & set(words), key=lambda k: list6.index(k))
            a.append(list12)
            # print(a)
            max(a) #the list with a max of matching words
            print(max(a))
            for i in range(6):
                if max(a) == a[i]: #for i bind with max list matched get it and then from the department list names take the name corresponding to it
                    print(hind[i])
                    global département_déplacement
                    département_déplacement = hind[i]
                    Noms = ['NICOLAS', 'KHADIJA', 'FRÉDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MÉDECIN', 'FATIHA', 'KAWTAR',
                            'HAJAR', 'KHATAB',
                            'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FRÉDÉRIC',
                            'NICO', 'SAAD','ORCHI'] #list of all the  emines' staff names
                    Noms1 = ['FRÉDÉRIC','NICOLAS', 'KHADIJA', 'KHATAB', 'SAAD', 'CHEIMANOFF', 'AITHADOUCH', 'NICO']  # direction
                    Noms2 = ['ZINEB', 'KHOUKH', 'ABDELAOUI', 'FATIHA', 'KAWTAR']  # scolarité
                    Noms3 = ['REDA', 'HAJAR', 'BOUCHIKHI', 'ORCHI']  # logistique
                    Noms4 = ['INFIRMIÈRE', 'MÉDECIN']  # health center
                    global NOMS
                    if département_déplacement == 'département direction':
                        NOMS = Noms1
                    elif département_déplacement == 'département de scolarité':
                        NOMS = Noms2
                    elif département_déplacement == 'département logistique':
                        NOMS = Noms3
                    elif département_déplacement == 'département santé':
                        NOMS = Noms4
                    term0 = set(words) & set(Noms)
                    str_val = " ".join(term0)
                    global term1
                    term1 = str_val
                    print(term1)
                    if term1 in words:  # see if one of the words in the sentence is the word we want
                        text_term1_say = "voulez vous allez au " + département_déplacement + " chez " + term1 #term1 is the name of the person we want
                        francais.speak(text_term1_say)
                        global s
                        s = 2
                        register()
                        hind_class.voice()
                    else:
                        text_term_say = "voulez vous allez au" + département_déplacement
                        francais.speak(text_term_say)
                        s=6
                        register()
                        hind_class.voice()
    def boucle1():
        words2 = text22.split()
        print(words2)
        words22 = [x.upper() for x in words2]  # new list of words with all the words majuscule
        print(words22)
        rsp = " ".join(words22)  # making the intersection words a string to use it
        if 'OUI' in rsp :
            print(NOMS)
            if term1 in NOMS:
                text__say = "d'accord, je vais vous guidez, s'il vous plait suivez moi"
                francais.speak(text__say)
                print("deplacer "+term1)
            else:
                text__say_else = "le nom et le département que vous voulez ne sont pas dépendants "
                francais.speak(text__say_else)
                hind_class.main_hind()
        elif 'NON' in rsp :
            global s
            s=1
            hind_class.main_hind()
        else:
            text__say1 = "je vous est pas bien compris, pouvez redire "
            francais.speak(text__say1)

            s = 2
            register()
            hind_class.voice()
    def boucle2():
        Text = text3.split()
        print(Text)
        words3 = [x.upper() for x in Text]  # new list of words with all the words majuscule
        print(words3)
        text33 = " ".join(words3)
        if 'OUI' in text33:
            text0 = "chez qui?"
            francais.speak(text0)
            global s
            s=4
            register()
            hind_class.voice()
        else:
            francais.speak("D'accord, suivez moi je vais vous emmenez au " + département_déplacement)
            print("D'accord, suivez moi je vais vous emmenez au " + département_déplacement)
            #geo(département_déplacement)
    def boucle3():
        words4 = text4.split()
        print(words4)
        z = [x.upper() for x in words4]
        Noms = ['NICOLAS', 'KHADIJA', 'FRÉDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MÉDECIN', 'FATIHA', 'KAWTAR',
                'HAJAR', 'KHATAB',
                'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FRÉDÉRIC',
                'NICO', 'SAAD']  # list of all the  emines' staff names
        term4 = set(z) & set(Noms)
        str_val = " ".join(term4)
        global term44
        term44 = str_val
        print(term44)
        if term44 in Noms:
            term41 = term44.lower()
            text_say3 = "voulez vous allez au" + département_déplacement + "chez " + term41
            francais.speak(text_say3)
            global s
            s=5
            register()
            hind_class.voice()
        else:
            francais.speak("je trouve pas la personne que vous voulez")
            hind_class.main_hind()
    def boucle4():
        word5 = text5.split()
        print(word5)
        word5 = [x.upper() for x in word5]  # new list of words with all the words majuscule
        print(word5)
        text55 = " ".join(word5)
        if 'OUI' in text55:
            print(NOMS)
            if term44 in NOMS:
                text_say_term44 = "d'accord, je vais vous guidez, s'il vous plait suivez moi"
                francais.speak(text_say_term44)
                print("deplacement"+term44)
                #geo(term44)
            else:
                text__say_else0 = "le nom et le département que vous voulez ne sont pas dans le même emplacement "
                francais.speak(text__say_else0)
                hind_class.main_hind()
        else:
            text0 = "je trouve pas la personne que vous voulez"
            francais.speak(text0)
            hind_class.main_hind()
    def boucle5():
        words1_oui = text6.split()
        print(words1_oui)
        print("hind a hind")
        words_oui = [x.upper() for x in words1_oui]  # new list of words with all the words majuscule
        print(words_oui)
        rsp = " ".join(words_oui)  # making the intersection words a string to use it
        if 'OUI' in rsp :
            text__say = "voulez vous allez chez une personne précise?"
            francais.speak(text__say)
            s = 3
            register()
            hind_class.voice()
        else:
            s=1
            hind_class.main_hind()
    #the function that changes the audio file contain to a text and calls the function
    #with all the possibilities to answer called boucle
    #def function voice; audio to text
    def voice():
        print('voicetotext')
        AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "C:\\filtered1.wav")
        # use the audio file as the audio source
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file
        # recognize speech using Google Speech Recognition
        
        try:
            TEXT = r.recognize_google(audio, language="fr-FR")
            print(TEXT)
            global s
            if s==0:
                text0 = TEXT
                print(text0)
                #taking the response from the visitor, the robot even start explaining and then runs the main function or just run it is
                #the visitor wanna pass the explanation
                Text0 = text0.split()
                list_continuer = ['continue','continuer','terminer','entendre']
                text001 = set(Text0) & set(list_continuer)  # setting the intersection between the two lists words and exit_list
                str_val = " ".join(text001)  # making the intersection words a string to use it
                text001 = str_val  # the new text variable we going to work with
                # two conditions whether the visitor wanna leave or complete
                try:
                    if text001 in list_continuer:
                        text01 = "d'accord, je commencerais :Au niveau du rez de chaussée se trouve cinq département; premièrement la réception qui se trouve au bloc A: " \
                                "si vous voulez mieux aitre université deuxièment la direction qui se trouve au bloc B: se trouve le bureau " \
                                "du directeur Nicolas Cheimanoff , son assistante Khadiija Aitahadouch et le bureau de Saad Aitkhatab le " \
                                "responsable de communication de EMINES troixièment la scolarité qui se trouve au bloc C: elle se trouve en face" \
                                " du foyer des élèves, se trouve le bureau de Fatiha Alabdelaoui responsable de scolarité de EMINES, Zineb " \
                                "Elkhoukh assistante du directeur d'éducation et de la recherche, et messieur Orchi responsable d'impression " \
                                "quatrièment la logistique qui se trouve au bloc D: il se trouve le bureau de Reda Elbouchikhi responsable " \
                                "hébergement son assistante hajar Azerkouk cinquièment le pôle santé: ou se trouve le médecin et les infirmières " \
                                "et dernièrement E-tech: le club de robotique Emines si vous voulez consulter les projets effectués par nos " \
                                "étudiants."
                        francais.speak(text01)
                        hind_class.main_hind()
                    else:
                        hind_class.main_hind()
                except sr.UnknownValueError:
                    text_say__0_try = "je vous ai pas bien entendu, pouvez vous répètez, merci"
                    francais.speak(text_say__0_try)
                    register()
                    hind_class.voice()

            elif s == 1:
                global text1
                text1 = TEXT
                print(text1)
                hind_class.boucle()
                
            elif s == 2:
                global text22
                text22 = TEXT
                print(text22)
                hind_class.boucle1()
            
            elif s==3:
                global text3
                text3 = TEXT
                print(text3)
                hind_class.boucle2()
                
            elif s==4:
                global text4
                text4 = TEXT
                print(text4)
                hind_class.boucle3()
                
            elif s==5:
                global text5
                text5 = TEXT
                print(text5)
                hind_class.boucle4()
                
            elif s==6:
                global text6
                text6 = TEXT
                print(text6)
                hind_class.boucle5()
        except sr.UnknownValueError:
            text_say__0 = "je vous ai pas bien entendu, pouvez vous répètez, merci"
            francais.speak(text_say__0)
            register()
            hind_class.voice()
        #the function that takes the voices from the microphone register it in a wav file and then filter it and call the function
        #that changes the audio file contain into a text called voice
    #def register; register voice and filter it
    #the main function here is the first one calles in the programm and it counts n the number of no responses once n is higher or
    #equal to 2 due to the bad quality of voice...etc, the visitor gonna be directed to tape his direction on an interface 
    #the main function; count n and execute the register function for n<=1 / n>=2 --> interface
    def main_hind():
        global n
        n += 1
        print(n)
        #if n <= 2:# n is the nombre of tries and said no as response
        text_main0 = "qu'elle est votre question?"
        francais.speak(text_main0)
        global s
        s=1
        print(s)
        register()
        hind_class.voice()
        # else:
        #     text_main1 = "je préfére que vous tapez votre question, afin que je puisse vous comprendre"

        #     francais.speak(text_main1)
            #ajouter une interface

    def start():
        #the starting text where the robot represente himself and offer to explain more about the school
        text00 = "Bonjour chers visiteurs je suis votre robot d'assistance, je suis destiné à vous aider à se déplacer au sein de " \
                "l'EMINES et vous diriger vers votre destination et aussi répondre à vos questions."
        text000="je commencera par vous décrire le plan de l'école pour que je puisse vous aider efficacement à se déplacer,      mais  vous pouvez dépasser " \
                "cette partie descriptive durant cette étape en disons je passe,   sinon et si vous voulez l'entendre disez simplement je continue, "
        #text0000="je vous renseigne aussi qu'une fois mon micro est ouvert pour vous entendre mon interface devienne verte."
        francais.speak(text00 +text000 )
        register()
        hind_class.voice()





class francais:
    engine = pyttsx3.init()
    fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
    engine.setProperty('voice', fr_voice_id)

    def there_exists(terms):
        global a
        for term in terms:
            if term in a:
                return True

    def speak(text):
        global clientsocket
        engine = pyttsx3.init()
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
        engine.setProperty('rate', 150)
        engine.setProperty('voice', fr_voice_id)
        #socket_send("nocart,speaktrue,")
        engine.say(text)
        engine.runAndWait()
        #socket_send("nocart,speakfalse,")

    def speak_register(texte):
        global a
        francais.speak(texte)
        register()
        AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "C:\\filtered1.wav")
        # use the audio file as the audio source
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file
        # recognize speech using Google Speech Recognition  
            try:
                a = r.recognize_google(audio, language="fr-FR")
                print(a)
            except:
                pass

        return a
    



    def choices(a):
        # BDD
        global answer,no,clientsocket
        Noms = ['Nicolas', 'Cheimanoff', 'Khadija', 'AitHadouch', 'Yvan',
                'Gaignebet', 'Bouchikhi', 'Reda', 'Fatiha', 'Elabdellaoui']
        jobs = ['directeur', 'logistique', 'responsable', 'recherche', 'mécatronique']
        l1 = ['Cheimanoff', 'Nicolas', 'Directeur', 'Nicolas.CHEIMANOFF@emines.um6p.ma', 'B']
        l2 = ['Gaignebet', 'Yvan', "Chargé d'enseignement de projets mécatroniques", 'Yvon.Gaignebet@emines.um6p.ma', 'D près des laboratoires']
        l3 = ['Elabdellaoui', 'Fatiha', 'Responsable de scolarite', 'Fatiha.ELABDELLAOUI@emines.um6p.ma', 'C']
        l4 = ['AitHadouch', 'Khadija', 'Assistante de direction', 'Khadija.AITHADOUCH@emines.um6p.ma', 'B']
        l5 = ['Bouchikhi', 'Reda', 'Responsable logistique', 'Reda.Bouchikhi@emines.um6p.ma', 'D']
        if francais.there_exists(Noms):
            no = no + 1
            print("aaa")
            list = a.split()
            term = set(list) & set(Noms)
            term0 = " ".join(term)
            if term0 in Noms:
                d=francais.speak_register("voulez vous savoir qui est " + term0)
                print(d)
                if 'oui' in d:
                    if term0 in ['Nicolas', 'Cheimanoff']:
                        l = l1
                    elif term0 in ['Gaignebet', 'Yvan']:
                        l = l2
                    elif term0 in ['Fatiha', 'Elabdellaoui']:
                        l = l3
                    elif term0 in ['Khadija', 'AitHadouch']:
                        l = l4
                    elif term0 in ['Reda', 'Bouchikhi']:
                        l = l5
                    la_liste="cart,"+str(l[1])+"," + str(l[0])+","+str(l[2])+","+str(l[3])+","+str(l[4])+","
                    print(la_liste)
                    #socket_send(la_liste.encode())
                    francais.speak(l[1] + l[0]+l[2] + "de l'EMINES  "+" son bureau se  trouve dans le bloc " + l[4]+" et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
        elif francais.there_exists(jobs):
            list1 = a.split()
            term1 = set(list1) & set(jobs)
            print(term1)
            job = " ".join(term1)
            list2 = job.split()
            list2.reverse()
            print(list2)
            no = no + 1
            if any(item in jobs for item in list2):
                d=francais.speak_register("voulez vous savoir qui est le " + job + "de l'EMINES")
                if 'oui' in d:
                    if any(item in ['directeur', "l'enseignement","enseignement"] for item in list2):
                        l = l1
                    if any(item in [ "responsable","logistique"] for item in list2):
                        l = l5
                    if any(item in ["projet", "mécatronique"] for item in list2):
                        l = l2
                    francais.speak(l[1] + l[0]+l[2] + "de l'EMINES  "+" son bureau se  trouve dans le bloc   " + l[4]+" et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                    la_la_liste="cart,"+str(l[1])+"," + str(l[0])+","+str(l[2])+","+str(l[3])+","+str(l[4])+","
                    #socket_send(la_la_liste.encode())
        elif francais.there_exists(["je veux visiter virtuellement l'université", "visite virtuelle"]):
            webbrowser.open_new_tab("https://alpharepgroup.com/um6p_univer/models.php")
            engine = pyttsx3.init()
            no = no + 1
            engine.say("Bienvenue à la visite virtuelle de um6p ")
            os.system("taskkill /f /im " )
            
        # time
        elif francais.there_exists(['quelle heure est-il', 'heure']):
            strTime = datetime.datetime.now().strftime("%H""heures""%M")
            engine = pyttsx3.init()
            engine.say(f"Il est {strTime}")
            no = no + 1


        # presentation
        elif francais.there_exists(['présente-toi', 'qui est tu']):
            engine = pyttsx3.init()
            no = no + 1
            engine.say("Je suis votre robot d'assistance" " Réalisé par les étudiants du quatrième année d'EMINE"
                  "Je suis un projet mécatronique pour rendre service aux visiteurs de l'université mohamed 6 polytechnique")
            
        # google
        elif francais.there_exists(['Google','google', 'ouvrir google']):
            webbrowser.open_new_tab("https://www.google.com")
            engine = pyttsx3.init()
            engine.say("Vous avez l'accés à Google")
            time.sleep(30)
            os.system("taskkill /f /im " )
            no = no + 1
            
        # localisation google maps
        elif francais.there_exists(["où suis-je exactement", "où suis-je", "où je suis ", "où je suis exactement", "localisation"]):
            webbrowser.open_new_tab("https://www.google.com/maps/search/Where+am+I+?/")
            engine = pyttsx3.init()
            engine.say("Selon Google maps, vous devez être quelque part près d'ici")
            time.sleep(5)
            os.system("taskkill /f /im " )
            no = no + 1
            
        # météo
        elif francais.there_exists(["météo", "combien fait-il de degrés maintenant", "degré"]):
            search_term = a.split("for")[-1]
            webbrowser.open_new_tab(
                "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5")
            engine = pyttsx3.init()
            engine.say("Voici ce que j'ai trouvé pour la météo sur Google")
            time.sleep(5)
            os.system("taskkill /f /im " )
            no = no + 1
            

        else:
            try:
                wolf(a)
            except:
                engine = pyttsx3.init()
                engine.say("Je vous ai pas compris")



    def start():

        global a,no
        no=0
        try:
            print("try")
            francais.speak_register("Quelle est votre question ?")
        except:
            print("pass")
            pass
        while True:
            if francais.there_exists(['pas vraiment', 'merci', 'bye', 'au revoir']):
                francais.speak("au revoir !")
                break
            francais.choices(a)
            if no>0:
                try:
                    francais.speak_register("Avez vous des questions ?")
                except:
                    pass
                if francais.there_exists(['oui', 'je veux continuer']):
                    try:
                        francais.speak_register("Quelle est votre question ?")
                    except:
                        pass
                if francais.there_exists(['non', 'pas vraiment', 'merci']):
                    francais.speak("au revoir !")
                    break
            if no==0:
                try:
                    francais.speak_register("Quelle est votre question ?")
                except:
                    pass

# server(8080)
# socket_send("no,no,no,no,no,no,")




# while True:
#     hand=0
#     socket_send("nocart,hand,")
#     while True:
#         isTrue, frame=vd.read()
#         #frame=framerealsense()
#         if handisdetected(frame)==True:
#             hand=hand+1
#             print(hand)
#         if hand==10:
#             hand=0
#             handisdetected(frame)==False
#             francais.speak("s'il vous plait, metez votre bavette")
#             socket_send("nocart,maskfalse,")
#             print("hand detected")
#             break
#     run()
#     socket_send("nocart,masktrue,")
#     francais.speak("merci pour mettre votre bavette")

#     while True:
#         socket_send("nocart,choix,")
#         time.sleep(1)
#         francais.speak("veuillez choisir une option")
#         while True:
#             commande=clientsocket.recv(1024).decode()
#             if commande=="deplacement" or commande=="question" or commande=="quitter":
#                 break
#         if commande=="deplacement":
#             hind_class.start()
#         elif commande=="question":
#             francais.start()
#         elif commande=="quitter":
#             francais.speak("merci pour votre visite, à la prochaine !")
#             commande=="nothing"
#             break
francais.start()

