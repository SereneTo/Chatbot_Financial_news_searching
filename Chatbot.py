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
         chatbot_response = chatbot_response + ''.join(data_copy['Headlines'][similar_sentence_number1]) + "    Sentiment: " + ''.join(data_copy['Sentiment'][similar_sentence_number1]) + '\n' + ''.join(data_copy['Description'][similar_sentence_number1]) + '\n\n' + ''.join(data_copy['Headlines'][similar_sentence_number2]) + "    Sentiment: " + ''.join(data_copy['Sentiment'][similar_sentence_number2]) + '\n' + ''.join(data_copy['Description'][similar_sentence_number2]) + '\n\n' + ''.join(data_copy['Headlines'][similar_sentence_number3]) + "    Sentiment: " + ''.join(data_copy['Sentiment'][similar_sentence_number3]) + '\n' + ''.join(data_copy['Description'][similar_sentence_number3])+ '\n\n '
         return chatbot_response


greeting_input_texts = ("hi", "hey", "heys", "hello", "morning", "evening", "greetings",)
greeting_replie_texts = ["hey", "hey hows you?", "*nods*", "hello there", "ello", "Welcome, how are you"]


def reply_greeting(text, greeting_input_texts, greeting_replie_texts):
    for word in text.split():
        if word.lower() in greeting_input_texts:
            return random.choice(greeting_replie_texts)


continue_discussion = True
print("Hello, I am a chatbot, I will answer your queries regarding financial news:")
while(continue_discussion == True):
    print("Do you want to search by headings or by content? (1 for heading and 2 for content)")
    user_input = input()
    correct_selection = False
    # print(correct_selection)
    while(correct_selection == False):
        if user_input == '1' or user_input == '2':
            search_type = user_input
            correct_selection = True
        elif(user_input == 'bye'):
            print("Chatbot: Take care, bye ..")
            quit()
        else:
            print("please input 1 or 2.")
            user_input = input()
    print("What do you want to know?")
    user_input = input()
    user_input = user_input.lower()
    if(user_input !='bye'):
        if(user_input =='thanks' or user_input =='thank you very much'  or user_input =='thank you'):
            continue_discussion=False
            print("Chatbot: Most welcome")
        else:
            if(reply_greeting(user_input, greeting_input_texts, greeting_replie_texts)!=None):
                print("Chatbot: "+reply_greeting(user_input, greeting_input_texts, greeting_replie_texts))
            else:
                print("Chatbot: ",end="")
                if(search_type == '1' ):
                    print(give_reply(user_input, sentence_headlines))
                    sentence_headlines.remove(user_input)
                else:
                    print(give_reply(user_input, sentence_description))
                    sentence_description.remove(user_input)
    else:
        continue_discussion=False
        print("Chatbot: Take care, bye ..")