import nltk.sentiment.vader as sev

sia = sev.SentimentIntensityAnalyzer()

def return_polarity_score(sentence):
        """
        :param sentence: sentence in string format
        :return: float value between -1.0 and 1.0
        """
        return sia.polarity_scores(sentence)["compound"]


if __name__ == "__main__":
    print(return_polarity_score("This is a negative message."))