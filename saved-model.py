import pandas as pd
import telebot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

dialogs = pd.read_csv('dialogs.csv')

vectorizer = TfidfVectorizer()
vectorizer.fit(dialogs.message)
matrix_big = vectorizer.transform(dialogs.message)

svd = TruncatedSVD(n_components=500)
svd.fit(matrix_big)
matrix_small = svd.transform(matrix_big)


def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)


class NeighborSampler(BaseEstimator):

    def __init__(self, k=5, temperature=1.0):
        self.k = k
        self.temperature = temperature

    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)

    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(X, return_distance=True,
                                              k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(np.random.choice(index,
                                           p=softmax(distance * self.temperature)))
        return self.y_[result]

ns = NeighborSampler()
ns.fit(matrix_small, dialogs.answer)
pipe = make_pipeline(vectorizer, svd, ns)

token = '1608570088:AAHGWdarXJrIlowm6shDywfWLOBdpGU6oUo'

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f'Привет, {message.from_user.first_name}')

@bot.message_handler(func=lambda message:True)
def send_message(message):
    bot.reply_to(message, pipe.predict([message.text.lower()])[0])

bot.polling()
