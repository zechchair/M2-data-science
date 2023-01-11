from gtts import gTTS
import os
from os import path 
import pyttsx3
from playsound import playsound 
lang='es'
def pytt3x_get():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        print("Voice: %s" % voice.name)
        print(" - ID: %s" % voice.id)
        print(" - Languages: %s" % voice.languages)
        print(" - Gender: %s" % voice.gender)
        print(" - Age: %s" % voice.age)
        print("\n")
from textblob import TextBlob


def translate(fromm,to_lang,text):
    print(TextBlob(text).detect_language())
    query=TextBlob(text).translate(from_lang=fromm,to=to_lang)
    return query
def say(text):#to talk in background without mouth moving 
    global lang
    engine = pyttsx3.init()
    if lang=="fr":
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
    elif lang=="en":
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0 "
    elif lang=='es':
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0 "

    
    engine.setProperty('rate', 150)
    engine.setProperty('voice', fr_voice_id)
    engine.say(text)
    engine.runAndWait()
def say_g(text):
    text=str(translate("en","ar",text))
    gTTS(text=text, lang='ar', slow=False) .save('eng.mp3')
    playsound('.\eng.mp3')  



say_g("hello")
#pytt3x_get()
