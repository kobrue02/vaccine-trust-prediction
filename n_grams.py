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


if __name__ == "__main__":
    print(n_gram("Hallo ich heiße Konrad und ich werde jetzt etwas essen.", 3))
