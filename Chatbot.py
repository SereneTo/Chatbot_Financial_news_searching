from FileLoader import *
from Preprocessing import *
import pandas as pd
import numpy as np
from FileLoader import *
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from textblob import TextBlob
import warnings
import tkinter
from tkinter import *


warnings.filterwarnings("ignore")

file = FileLoader("reuters_headlines.csv")
data = file.read_file()
data.insert(data.shape[1], 'Sentiment', 0)

for i in range(len(data)):
    corpus = TextBlob(data['Headlines'][i] + ' ' + data['Description'][i])
    if(corpus.sentiment.polarity > 0):
        data['Sentiment'][i] = "Positive"
    elif(corpus.sentiment.polarity < 0):
        data['Sentiment'][i] = "Negative"
    else:
        data['Sentiment'][i] = "Neutral"

# print(data.info())
# print(data.Sentiment.value_counts())
data_copy = data.copy()
preprocessed_data = Preprocessing(data)
preprocessed_data.error_cleaning("Headlines")
preprocessed_data.error_cleaning("Description")

headlines = preprocessed_data.sentence_normalizatio("Headlines")
description = preprocessed_data.sentence_normalizatio("Description")

def convert_token_to_string(text):
    return " ".join(text)

sentence_headlines = headlines.apply(convert_token_to_string)
sentence_headlines = [sentence for sentence in sentence_headlines]
sentence_description = description.apply(convert_token_to_string)
sentence_description = [sentence for sentence in sentence_description]
# dict_headlines = dict(zip(sentence_headlines,[i for i in range(len(sentence_headlines))]))
# dict_description = dict(zip(sentence_description,[i for i in range(len(sentence_description))]))

def give_reply(user_input, sentence_list):
     chatbot_response=''
     sentence_list.append(user_input)
     word_vectors = TfidfVectorizer()
     vectorized_words = word_vectors.fit_transform(sentence_list)
     similarity_values = cosine_similarity(vectorized_words[-1], vectorized_words)
     similar_sentence_number1 =similarity_values.argsort()[0][-2]
     similar_sentence_number2 = similarity_values.argsort()[0][-3]
     similar_sentence_number3 = similarity_values.argsort()[0][-4]
     # print("similar_sentence_number",similar_sentence_number1,similar_sentence_number2,similar_sentence_number3)
     similar_vectors = similarity_values.flatten()
     similar_vectors.sort()
     matched_vector = similar_vectors[-2]
     if(matched_vector == 0):
         chatbot_response = chatbot_response+"I am sorry! I don't understand you"
         return chatbot_response
     else:
         chatbot_response = chatbot_response + str(data_copy['Headlines'][similar_sentence_number1]) + "\nSentiment: " + str(data_copy['Sentiment'][similar_sentence_number1]) + '\n' + str(data_copy['Description'][similar_sentence_number1]) + '\n\n' + str(data_copy['Headlines'][similar_sentence_number2]) + "\nSentiment: " + str(data_copy['Sentiment'][similar_sentence_number2]) + '\n' + str(data_copy['Description'][similar_sentence_number2]) + '\n\n' + str(data_copy['Headlines'][similar_sentence_number3]) + "\nSentiment: " + str(data_copy['Sentiment'][similar_sentence_number3]) + '\n' + str(data_copy['Description'][similar_sentence_number3])+ '\n\n '

         # chatbot_response = chatbot_response + ''.join(data_copy['Headlines'][similar_sentence_number1]) + "    Sentiment: " + ''.join(data_copy['Sentiment'][similar_sentence_number1]) + '\n' + ''.join(data_copy['Description'][similar_sentence_number1]) + '\n\n' + ''.join(data_copy['Headlines'][similar_sentence_number2]) + "    Sentiment: " + ''.join(data_copy['Sentiment'][similar_sentence_number2]) + '\n' + ''.join(data_copy['Description'][similar_sentence_number2]) + '\n\n' + ''.join(data_copy['Headlines'][similar_sentence_number3]) + "    Sentiment: " + ''.join(data_copy['Sentiment'][similar_sentence_number3]) + '\n' + ''.join(data_copy['Description'][similar_sentence_number3])+ '\n\n '
         return chatbot_response


greeting_input_texts = ("hi", "hey", "heys", "hello", "morning", "evening", "greetings",)
greeting_replie_texts = ["hey", "hey hows you?", "*nods*", "hello there", "ello", "Welcome, how are you"]


def reply_greeting(text, greeting_input_texts, greeting_replie_texts):
    for word in text.split():
        if word.lower() in greeting_input_texts:
            return random.choice(greeting_replie_texts)


# # with Console
# continue_discussion = True
# print("Hello, I am a chatbot, I will answer your queries regarding financial news:")
# while(continue_discussion == True):
#     print("Do you want to search by headings or by content? (1 for heading and 2 for content)")
#     user_input = input()
#     correct_selection = False
#     # print(correct_selection)
#     while(correct_selection == False):
#         if user_input == '1' or user_input == '2':
#             search_type = user_input
#             correct_selection = True
#         elif(user_input == 'bye'):
#             print("Chatbot: Take care, bye ..")
#             quit()
#         else:
#             print("please input 1 or 2.")
#             user_input = input()
#     print("What do you want to know?")
#     user_input = input()
#     user_input = user_input.lower()
#     if(user_input !='bye'):
#         if(user_input =='thanks' or user_input =='thank you very much'  or user_input =='thank you'):
#             continue_discussion=False
#             print("Chatbot: Most welcome")
#         else:
#             if(reply_greeting(user_input, greeting_input_texts, greeting_replie_texts)!=None):
#                 print("Chatbot: "+reply_greeting(user_input, greeting_input_texts, greeting_replie_texts))
#             else:
#                 print("Chatbot: ",end="")
#                 if(search_type == '1' ):
#                     print(give_reply(user_input, sentence_headlines))
#                     sentence_headlines.remove(user_input)
#                 else:
#                     print(give_reply(user_input, sentence_description))
#                     sentence_description.remove(user_input)
#     else:
#         continue_discussion=False
#         print("Chatbot: Take care, bye ..")



correct_input = False
search_type = 0
def getResponse(user_input):
    global correct_input
    global search_type

    if (user_input != 'bye'):
        if (user_input == 'thanks' or user_input == 'thank you very much' or user_input == 'thank you'):
            quit()
        elif reply_greeting(user_input, greeting_input_texts, greeting_replie_texts)!=None:
            response = reply_greeting(user_input, greeting_input_texts, greeting_replie_texts)
            return response

        else:
            if correct_input == False:
                if user_input == '1' or user_input == '2':
                    search_type = int(user_input)
                    correct_input = True
                    response = "What do you want to know?"
                    return response

                else:
                    response = "please input 1 or 2."
                    search_type = 0
                    return response
            else:
                if search_type == 1:
                    user_input = user_input.lower()
                    response = give_reply(user_input, sentence_headlines)
                    sentence_headlines.remove(user_input)
                    search_type = 0
                    correct_input = False
                    return response
                elif search_type == 2:
                    user_input = user_input.lower()
                    response = give_reply(user_input, sentence_description)
                    sentence_description.remove(user_input)
                    search_type = 0
                    correct_input = False
                    return response
                else:
                    response = "Something wrong happened."
                    search_type = 0
                    correct_input = False
                    return response
    else:
        quit()


# with GUI
base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.insert(END, "Bot: " + "\nHello, I am a chatbot, I will answer your queries regarding financial news. Do you want to search by headings or by content? (1 for heading and 2 for content)" + '\n\n')
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + '\n' + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        res = getResponse(msg)
        ChatLog.insert(END, "Bot: " + '\n' + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="green", activebackground="#3c9d9b",fg='black',
                    command= send )

scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()

