import nltk

def n_gram(sent, n):
    grams = []
    tokens = nltk.word_tokenize(sent)
    for i in range(len(tokens)-n+1):
        gram = tokens[i]
        for j in range(1,n):
            gram += f" {tokens[i+j]}"
        grams.append(gram)
    return grams


def most_common_ngrams(corpus: list, n):
    bigram_frequency = {}
    bigrams = [n_gram(sent, n) for sent in corpus]
    bigrams = flatten(bigrams)
    for bigram in bigrams:
        if bigram not in bigram_frequency:
            bigram_frequency[bigram] = 1
        else:
            bigram_frequency[bigram] += 1
    top_keys = sorted(bigram_frequency, key=bigram_frequency.get, reverse=True)[:5]
    return top_keys

def flatten(l):
    return [item for sublist in l for item in sublist]



if __name__ == "__main__":
    print(n_gram("Hallo ich heiße Konrad und ich werde jetzt etwas essen.", 3))
