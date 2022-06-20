import imp
import pickle
import re
import string
import nltk as nltk
import pandas as pd
import json
from flask import Flask, url_for, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import jsonify
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pprint
import sklearn
from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import *
from flask_cors import *
import io
import base64

from wordcloud import STOPWORDS, WordCloud


vect, clf = pickle.load(open('model.pkl', 'rb'))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

@app.route('/wordcloud', methods=['POST'])
def get_word_cloud():
    if request.method == "POST":
        text = request.json.get("word")
        max_words = request.json.get("max_words")
        width=request.json.get("width")
        height=request.json.get("height")
        print(text)
        print(max_words)
        pil_img = WordCloud(
            max_words=max_words,
            width=width,
            stopwords=STOPWORDS,
            height=height, 
            background_color="white",colormap='inferno').generate(text=text).to_image()
        img = io.BytesIO()
        pil_img.save(img, "PNG")
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        return img_base64


@app.route('/')
def root():
    return "Hello World"


def text_process(mess):
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    review = re.sub('[^a-zA-Z]', ' ', no_punc)
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    return review


def remove_stopwords_spam(text):
    stop_words = stopwords.words("english") + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text


@app.route('/spam', methods=['POST'])
def spam():
    if request.method == "POST":
        json_data = json.loads(request.data)
        print(json_data)
        # print(json_data["comments"])
        data = pd.DataFrame(json_data)
        x_test = data['comments']
        test_transform = vect.transform(x_test)
        predicted = clf.predict(test_transform)
        data["pred"] = predicted
        result = data["pred"].map({0: 'ham', 1: 'spam'}).to_list()
        return json.dumps(result)


def processed_data(text, stop_remove, more_remove, stem, lemme):
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(w) for w in text]
    text = ("".join(text))  # joining of the text .
    print(text)
    return text



def sentence_score(text):
    sid = SentimentIntensityAnalyzer()  # making the object of the sentimentanalyser
    sid_dict = sid.polarity_scores(text)  # getting the scores ductionary
    scores = sid_dict['compound']  # taking the compound scores
    print(sid_dict)
    if (scores >= 0.05):  # based on the compound score finding the sentiments of the review
        return "positive"
    elif (scores <= -0.05):
        return "negative"
    else:
        return "netural"


@app.route('/sentiment', methods=['POST'])
def semtiment():
    if request.method == "POST":
        json_data = json.loads(request.data)
        df = pd.DataFrame(json_data)
        df.dropna(subset=['comments'], inplace=True)
        df['processed'] = df['comments'].apply(processed_data, args=(False, True, False, True))
        df['sentiments'] = df['processed'].apply(sentence_score)
        return jsonify(df['sentiments'].to_list())



def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


@app.route('/topics', methods=['POST'])
def topics():
    if request.method == "POST":
        json_data = json.loads(request.data)
        df = pd.DataFrame(json_data)
        df.dropna(subset=['comments'], inplace=True)
        df['processed'] = df['comments']
        review = df.processed.values.tolist()
        data_words = list(sent_to_words(review))
        data_words = remove_stopwords(data_words)

        # Create Dictionary
        id2word = corpora.Dictionary(data_words)

        # Create Corpus
        texts = data_words

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # remove stop words
        data_words = remove_stopwords(data_words)

        # # number of topics
        num_topics = 10

        # Build LDA model
        lda_model20 = gensim.models.LdaMulticore(corpus=corpus,
                                                 id2word=id2word,
                                                 num_topics=num_topics)

        topics = pprint.pformat(lda_model20.show_topics())

        print(topics)
        return topics


if __name__ == "__main__":
    app.run(debug=True) 
