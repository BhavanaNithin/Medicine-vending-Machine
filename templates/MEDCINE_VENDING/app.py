from flask import Flask, render_template, request, redirect, url_for
import os
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import sqlite3
import telepot
from pprint import pprint
import time
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
from serial_test import Send
def TTS(text1):
    print(text1)
    if str(text1) in 'Dextromethorphan' or str(text1) in 'Guaifenesin' or str(text1) in 'Albuterol'  or str(text1) in 'Amoxicillin' :
        Send('A')
        print('cough tablets')
        myobj = gTTS(text="You have a cough, please collect the medicine", lang='en-IN', tld='com', slow=False)
        myobj.save("voice.mp3")
        print('\n------------Playing--------------\n')
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
        time.sleep(1)

    if str(text1) in 'Paracetamol (Acetaminophen)' or str(text1) in 'Aspirin' or str(text1) in 'fever' :
        Send('B')
        print('fever tablets')
        myobj = gTTS(text="You have a fever, please collect the medicine", lang='en-IN', tld='com', slow=False)
        myobj.save("voice.mp3")
        print('\n------------Playing--------------\n')
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
        time.sleep(1)
    if str(text1) in 'Antihistamines' or str(text1) in 'Cough Suppressants' or str(text1) in 'cold'  :
        Send('C')
        print('cold')
        myobj = gTTS(text="You have a cold, please collect the medicine", lang='en-IN', tld='com', slow=False)
        myobj.save("voice.mp3")
        print('\n------------Playing--------------\n')
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
        time.sleep(1)
    if str(text1) in 'Dextromethorphan' or str(text1) in 'Guaifenesin' or str(text1) in 'Albuterol'  or str(text1) in 'headache' :
        Send('D')   
        print('headache....')
        myobj = gTTS(text="You have a headache, please collect the medicine", lang='en-IN', tld='com', slow=False)
        myobj.save("voice.mp3")
        print('\n------------Playing--------------\n')
        song = MP3("voice.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load('voice.mp3')
        pygame.mixer.music.play()
        time.sleep(song.info.length)
        pygame.quit()
        time.sleep(1)    
    myobj = gTTS(text=text1, lang='en-IN', tld='com', slow=False)
    myobj.save("voice.mp3")
    print('\n------------Playing--------------\n')
    song = MP3("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()
    time.sleep(1)

# Load prediction model
stemmer = LancasterStemmer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def getResponse(ints, intents_json):
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            global result
            result = random.choice(i['responses'])
            break
    return result

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__)
chat_history = []

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/home')
def home():
   return render_template("index.html")

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        user_input = request.form['query']
        output = chatbot_response(user_input)
        chat_history.append([user_input, output])
        TTS(output)
        
        return render_template('chatbot.html', chat_history=chat_history)
    return render_template('chatbot.html')
   
if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)
