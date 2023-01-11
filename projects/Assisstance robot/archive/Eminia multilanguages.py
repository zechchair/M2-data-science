
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
import argparse
import cv2
import numpy as np
from yolo import YOLO
import serial
# ser = serial.Serial('COM18', baudrate = 9600, timeout = 1)
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
from playsound import playsound
from gtts import gTTS
import time
import pyrealsense2 as rs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
r = sr.Recognizer()
browserExe = "chrome.exe"
list_max = []
Noms = []
s = 0  # the number that helps to know when excuting the voice function what i should excute the main boucle or a boucle inside it
start_time = 0
n = 0
no = 0
wolf_text = ""
lang = 'fr'

client = wolframalpha.Client('QXGQJX-K97LHHGQ3L')
hand = 0
vd = cv2.VideoCapture(0)
# isTrue, frame=vd.read()


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

# load mask model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


def wolf(query):
    global lang, wolf_text

    query1 = TextBlob(query).translate(to='en')
    # print(query1)
    res = client.query(query1)
    output = next(res.results).text
    # print(output)
    output2 = TextBlob(output).translate(to='fr')
    if output2 == "Je m'appelle Wolfram | Alpha." or output2 == "Je m'appelle Wolfram | Alpha":
        output2 = "Je m'appelle Eminia."
    print(output2)
    speak(str(output2))


def mail_send(mail_content):
    # The mail addresses and password
    sender_address = 'eminia.robot@gmail.com'
    sender_pass = 'EMINIA2020'
    receiver_address = 'ayoub.talibi@emines.um6p.ma'
    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Eminia.'  # The subject linee
    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')


def battery_level():
    while True:
        b = str(8888) + "\n"
        ser.write(b.encode())
        data = ser.readline().decode('ascii')

        if data != "":
            battery = int(data)
            # print(battery)
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
    mask = False
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        isTrue, frame = vd.read()
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
                mask = False
                socket_send("maskfalse,")
                say("s'il vous plait, mettez votre masque")
            elif (mask > 0.90):
                print("mask")
                mask = True
                socket_send("masktrue,")
                say("Merci d'avoir porter votre masque")

        if mask == True:
            print("here")
            mask = False
            break


def server(Port):
    global clientsocket, server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Creates an instance of socket
    maxections = 999
    IP = '127.0.0.1'  # IP address of local machine
    server.settimeout(100)
    server.bind(('', Port))
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
    # charge=int(battery_level())
    charge = 40
    # if charge < 20:
    #     mail_send("my battery is low, please charge me")
    to_send = str(charge) + "," + text
    while True:
        clientsocket.send(to_send.encode())

        if clientsocket.recv(1024).decode() == "rcvd":
            break


def register():#to register the audio
        print('register')
        freq = 41000
        #i should calculate the duration using the volume
        duration = 6
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        write("test1.wav", freq, recording)
        fname = 'test1.wav'
        outname = 'filtered1.wav'
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



def say(text):  # to talk in background without mouth moving
    global lang
    if lang == "fr":
        engine = pyttsx3.init()
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
        engine.setProperty('rate', 150)
        engine.setProperty('voice', fr_voice_id)
        engine.say(text)
        engine.runAndWait()
    else:
        text = translate("fr", lang, text)
        gTTS(text=text, lang=lang, slow=False).save("eng.mp3")
        playsound('.\eng.mp3')
        os.remove("eng.mp3")


def translate(fromm, to_lang, text):
    query = TextBlob(text).translate(from_lang=fromm, to=to_lang)
    return str(query)


def speak(text):
    global clientsocket, lang

    if lang == "fr":
        engine = pyttsx3.init()
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
        # elif lang=="en":
        #     text=translate("fr","en",text)
        #     fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0 "
        # elif lang=="es":
        #     text=translate("fr","es",text)
        #     fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0 "
        engine.setProperty('rate', 150)
        engine.setProperty('voice', fr_voice_id)
        socket_send("speak,speaktrue,")
        engine.say(text)
        engine.runAndWait()
        socket_send("speak,speakfalse,")
    else:

        text = str(translate("fr", lang, text))
        print(text)
        socket_send("speak,speaktrue,")
        gTTS(text=text, lang=lang, slow=False).save("eng.mp3")
        playsound('.\eng.mp3')
        os.remove("eng.mp3")
        socket_send("speak,speakfalse,")


def speak_register(texte):
    global lang, wolf_text
    speak(texte)
    text = ""
    while True:
        try:
            register()
            AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "test1.wav")
            # use the audio file as the audio source
            r = sr.Recognizer()
            with sr.AudioFile(AUDIO_FILE) as source:
                audio = r.record(source)  # read the entire audio file
                # recognize speech using Google Speech Recognition
                if lang == "fr":
                    text = r.recognize_google(audio, language="fr")
                else:
                    wolf_text = r.recognize_google(audio, language=lang)
                    text = translate(lang, "fr", wolf_text)
                break

        except:
            speak("je vous ai pas bien entendu")

    text = text.upper()
    print(text)
    return str(text)


class francais:

    def there_exists(terms, speech):
        for term in terms:
            term = term.upper()
            if term in speech:
                return True

    def choices(speech):
        # BDD
        global no, clientsocket
        Noms = ['NICOLAS', 'CHEIMANNOF', 'KHADIJA', 'AITHADOUCH', 'YVAN', 'IVAN', 'YVON'
                                                                                  'GAIGNEBET', 'BOUCHIKHI', 'REDA',
                'FATIHA', 'ELABDLAOUI', 'NICOLA', 'RIDA']
        jobs = ['DIRECTEUR', 'RESPONSABLE LOGISTIQUE', 'LOGISTIQUE', 'RESPONSABLE DE LA LOGISTIQUE', 'RECHERCHE',
                'MECATRONIQUE']
        l1 = ['Cheimanoff', 'Nicolas', 'Directeur', 'Nicolas.CHEIMANOFF@emines.um6p.ma', 'B']
        l2 = ['Gaignebet', 'Yvon', "responsable de projets", 'Yvon.Gaignebet@emines.um6p.ma', 'D près des laboratoires']
        l3 = ['Elabdellaoui', 'Fatiha', 'Responsable de scolarite', 'Fatiha.ELABDELLAOUI@emines.um6p.ma', 'C']
        l4 = ['AitHadouch', 'Khadija', 'Assistante de direction', 'Khadija.AITHADOUCH@emines.um6p.ma', 'B']
        l5 = ['Bouchikhi', 'Reda', 'Responsable logistique', 'Reda.Bouchikhi@emines.um6p.ma', 'D']
        if francais.there_exists(Noms, speech):
            no = 1
            list = speech.split()
            term = set(list) & set(Noms)
            term0 = " ".join(term)
            if term0 in Noms:
                if term0 in ['NICOLAS', 'CHEIMANNOF', 'NICOLA']:
                    l = l1
                elif term0 in ['GAIGNEBET', 'YVAN', 'IVAN', 'YVON']:
                    l = l2
                elif term0 in ['FATIHA', 'ELABDLAOUI']:
                    l = l3
                elif term0 in ['KHADIJA', 'AITHADOUCH']:
                    l = l4
                elif term0 in ['REDA', 'BOUCHIKHI', 'RIDA']:
                    l = l5

                while True:
                    answer = speak_register("voulez vous savoir qui est " + l[0] + l[1])
                    if francais.there_exists(['OUI', 'ouais'], answer):

                        la_liste = "cart," + str(l[1]) + "," + str(l[0]) + "," + str(l[2]) + "," + str(
                            l[3]) + "," + str(l[4]) + ","

                        speak(l[1] + l[0] + l[2] + "de l'EMINES  " + " son bureau se  trouve dans le bloc " + l[
                            4] + " et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                        socket_send(la_liste)
                        print(la_liste)
                        while True:
                            resp = speak_register("est-ce que vous voulez allez chez " + l[1] + " ?")
                            if francais.there_exists(["OUI", "EFFECTIVEMENT", "BIEN SUR", 'ouais'], resp):
                                speak("suiver moi")
                                break
                            elif francais.there_exists(["NON", "PAS", "PROCHAINE FOIS", 'À'], resp):
                                break
                        break

                    elif francais.there_exists(['pas', 'non', 'A'], answer):
                        break
        elif francais.there_exists(jobs, speech):
            list1 = speech.split()
            term1 = set(list1) & set(jobs)
            job = " ".join(term1)
            list2 = job.split()
            no = 1
            if any(item in jobs for item in list2):
                while True:
                    if any(item in ['DIRECTEUR'] for item in list2):
                        l = l1
                    if any(item in ["RESPONSABLE", "LOGISTIQUE"] for item in list2):
                        l = l5
                    if any(item in ["RESPONSABLE", "MECATRONIQUE"] for item in list2):
                        l = l2
                    n_answer = speak_register("voulez vous savoir qui est le " + l[2] + "de l'EMINES")
                    if francais.there_exists(['OUI', 'ouais'], answer):
                        speak(l[1] + l[0] + l[2] + "de l'EMINES  " + " son bureau se  trouve dans le bloc   " + l[
                            4] + " et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                        la_la_liste = "cart," + str(l[1]) + "," + str(l[0]) + "," + str(l[2]) + "," + str(
                            l[3]) + "," + str(l[4]) + ","
                        socket_send(la_la_liste)
                        print(la_la_liste)

                        while True:
                            resp = speak_register("est-ce que vous voulez allez chez " + l[1] + " ?")
                            if francais.there_exists(["OUI", "EFFECTIVEMENT", "BIEN SUR", 'ouais'], resp):
                                speak("suiver moi")
                                break
                            elif francais.there_exists(["NON", "PAS", "PROCHAINE FOIS", 'À'], resp):
                                break
                        break
                    elif francais.there_exists(["non", 'pas', 'À'], answer):
                        break
        elif francais.there_exists(["je veux visiter virtuellement l'université", "visite virtuelle"], speech):
            webbrowser.open_new_tab("https://alpharepgroup.com/um6p_univer/models.php")
            engine = pyttsx3.init()
            no = 1
            engine.say("Bienvenue à la visite virtuelle de um6p ")
            os.system("taskkill /f /im ")
        elif francais.there_exists(["présente-moi", "projet", "projets"], speech):
            print("qrcode")

        # time
        elif francais.there_exists(['quelle heure est-il', 'heure'], speech):
            strTime = datetime.datetime.now().strftime("%H""heures""%M")
            engine = pyttsx3.init()
            engine.say(f"Il est {strTime}")
            no = 1


        # presentation
        elif francais.there_exists(['présente-toi', 'qui est tu'], speech):
            engine = pyttsx3.init()
            no = 1
            engine.say("Je suis votre robot d'assistance" " Réalisé par les étudiants du quatrième année d'EMINE"
                       "Je suis un projet mécatronique pour rendre service aux visiteurs de l'université mohamed 6 polytechnique")

        # google
        elif francais.there_exists(['Google', 'google', 'ouvrir google'], speech):
            webbrowser.open_new_tab("https://www.google.com")
            engine = pyttsx3.init()
            engine.say("Vous avez l'accés à Google")
            time.sleep(30)
            os.system("taskkill /f /im ")
            no = 1

        # localisation google maps
        elif francais.there_exists(
                ["où suis-je exactement", "où suis-je", "où je suis ", "où je suis exactement", "localisation"],
                speech):
            webbrowser.open_new_tab("https://www.google.com/maps/search/Where+am+I+?/")
            engine = pyttsx3.init()
            engine.say("Selon Google maps, vous devez être quelque part près d'ici")
            time.sleep(5)
            os.system("taskkill /f /im ")
            no = 1

        # météo
        elif francais.there_exists(["météo", "combien fait-il de degrés maintenant", "degré"], speech):
            search_term = a.split("for")[-1]
            webbrowser.open_new_tab(
                "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5")
            engine = pyttsx3.init()
            engine.say("Voici ce que j'ai trouvé pour la météo sur Google")
            time.sleep(5)
            os.system("taskkill /f /im ")
            no = 1


        else:
            try:
                wolf(speech)
            except:
                engine = pyttsx3.init()
                engine.say("Je vous ai pas compris")

    def start():
        global no
        no = 0
        speech = speak_register("Quelle est votre question")
        while True:
            if francais.there_exists(
                    ['bye', 'a bientot', "a toute", 'non', 'pas vraiment', 'au revoir', "quitter", 'fini', "c'est tout",
                     "je n'ai pas", "IL N'A AUCUNE QUESTION"], speech):
                speak("au revoir !")
                break
            francais.choices(speech)
            if no > 0:
                speech = speak_register("si tu as des autres question, n'hesitez pas ")
            if no == 0:
                speech = speak_register("Quelle est votre question ?")


server(8080)
while True:
    clientsocket.send("40,no,no,no,no,no,no,".encode())
    if clientsocket.recv(1024).decode() == "rcvd":
        break

while True:
    hand = 0
    socket_send("yellow,")
    socket_send("hand,")
    while True:
        isTrue, frame = vd.read()
        # frame=framerealsense()
        if handisdetected(frame) == True:
            hand = hand + 1
            print(hand)
        if hand == 10:
            hand = 0
            handisdetected(frame) == False
            print("hand detected")
            break

    run()

    socket_send("dq,")
    say(
        "Je me présente, Je suis EMINIA, votre robot d'assistance, réalisé dans le cadre des projets mécatroniques du deuxième année cycle d'ingénieurs à l EMINES.")
    socket_send("dq,cv,")
    say("Je peux vous présenter une personne du staff de l'école")
    socket_send("dq,dep,")
    say( "vous guider jusqu'à son bureau" )
    socket_send("dq,qes,")
    say("répondre à tout type de questions générales" )
    socket_send("dq,code,")
    say("ou vous faire un tour dans le forum pour vous présenter les différents projets ")
    socket_send("lang,")
    socket_send("hide,")
    # try:
    #     say("veuillez choisir la langue que vous preferez?")
    #     clientsocket.settimeout(5)
    #     lang = clientsocket.recv(1024).decode()
    # except:
    #     socket_send('hidelang,')
    while True:
        l = speak_register("veuillez dire quelle langue vous preferez")
        if 'FRANCAIS' in l:
            lang = 'fr'
            socket_send("fr,")
            say("Vous avez choisi le francais")
            break
        elif 'ANGLAIS' in l:
            lang = 'en'
            socket_send("en,")
            text="Vous avez choisi l'anglais"
            say(text)
            break
        elif 'ARABE' in l:
            lang = 'ar'
            socket_send("ar,")
            text="Tu as choisi l'arabe"
            say(text)
            break
    socket_send('hidelang,')
