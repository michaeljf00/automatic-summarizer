#!/usr/bin/env python
# coding: utf-8
import nltk
import os
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from xml.dom.minidom import parse

import numpy as np
import networkx as nx


def read_document(document):
    sentences = []

    contents = document.split('.')

    for sentence in contents:
        #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences



def readData():

    ret = dict()

    directory = "ScisummNet/scisummnet_release1.1__20190413/top1000_complete/"
    for doc_dir in os.listdir(directory):
        file_path = os.path.join(directory, doc_dir) + "/Documents_xml/" + doc_dir + ".xml"


        file = parse(file_path)
        models = file.getElementsByTagName('S')

        #print(doc_dir)

        # Find first line in XML file that contains data
        i = 0
        while(not(models[i].firstChild)): i += 1
        title = models.pop(i).firstChild.data

        # Store in dictionary with key = first occurring string and value = contents of file
        ret[title] = str()

        len_models = len(models)
        while(i < len_models):
            if (models[i].firstChild): ret[title] += models[i].firstChild.data
            i += 1

    return ret


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(document, top_n=5):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = list()

    # Step 1 - Read text anc split it
    sentences = read_document(document)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize text
    print("Summarize Text: \n", ". ".join(summarize_text))

# let's begin

if __name__ == "__main__":
    data = readData()
    for title, content in data.items():
        print(title)
        generate_summary(content)
        print()
