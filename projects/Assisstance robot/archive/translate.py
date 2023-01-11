from textblob import TextBlob


def translate(to_lang,text):
    print(TextBlob(text).detect_language())
    query=TextBlob(text).translate(to=to_lang)
    return query

while True:

    to=input("to language: ")
    text=input("text : ")

    print(translate(to,text))