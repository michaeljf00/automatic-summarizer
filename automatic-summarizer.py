#!/usr/bin/env python
# coding: utf-8
import os

from math import sqrt
import numpy as np

from nltk.corpus import stopwords
from xml.dom.minidom import parse

import networkx as nx

from pyrouge import Rouge155

r = Rouge155()
r.system_dir = 'path/to/system_summaries'
r.model_dir = 'path/to/model_summaries'
r.system_filename_pattern = 'some_name.(\d+).txt'
r.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)

def evaluateRougeScore(documnent):
    pass

def readDocument(document):
    sentences = list()

    contents = document.split('.')

    for sentence in contents:
        # Refine sentence and replace sentences that begin with non-alpha characters
        refinedSentence = sentence.replace("[^a-zA-Z]", " ").split(" ")
        # Add refined sentence to list of sentences
        sentences.append(refinedSentence)
    sentences.pop()

    # Filter out sentences that contain only two or less words
    sentences = list(filter(lambda x: len(x) > 2, sentences))

    #print(sentences)

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


# Calculate the cosine distance between two lists
def cosSimilarity(l1, l2):
    # Follows the formula for cosine distance 1 - (u.v / |u||v|)
    return 1 - (np.dot(l1, l2) / (sqrt(np.dot(l1, l1)) * sqrt(np.dot(l2, l2))))


def sentenceSimilarity(s1, s2, stopwords=None):
    if stopwords is None: stopwords = list()

    s1 = [t.lower() for t in s1]
    s2 = [t.lower() for t in s2]

    # Combined set of words from both sentences
    words = list(set(s1 + s2))

    wordsLength = len(words)

    # Initialize vectors for both setences
    l1 = [0] * wordsLength
    l2 = [0] * wordsLength

    # Create vector for the first sentence
    for w in s1:
        if w in stopwords: continue
        l1[words.index(w)] += 1

    # Create the vector for the second sentence
    for w in s2:
        if w in stopwords: continue
        l2[words.index(w)] += 1

    return 1 - cosSimilarity(l1, l2)


def createSimilarityMatrix(sentences, stop_words):
    # Create an empty similarity matrix
    sentenceLength = len(sentences)
    similarityMatrix = np.zeros((sentenceLength, sentenceLength))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:  continue # ignore if both are same sentences 
            similarityMatrix[i][j] = sentenceSimilarity(sentences[i], sentences[j], stop_words)

    return similarityMatrix


# Summarize the given document
def summarize(document, n=10):
    stopWords = stopwords.words('english')
    topSentences = list()

    # Split document into a list of sentences
    sentences = readDocument(document)

    # Create a similarity matrix in regards to all of the sentences in the document
    similarityMartix = createSimilarityMatrix(sentences, stopWords)

    # Rank sentences in similarity martix
    similarityGraph = nx.from_numpy_array(similarityMartix)
    scores = nx.pagerank(similarityGraph, max_iter=100)

    # Sort by sentance rank
    rankedSentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    for i in range(n): topSentences.append(" ".join(rankedSentences[i][1]))

    print("Text Summmary: \n", ". ".join(topSentences) + '.\n\n')


# Get summaries of the documents in the dataset
def getSummaries(n=1):
    data = readData()
    i = 0
    for title, content in data.items():
        print(title)
        summarize(content)
        i += 1
        if (i == n): break


def main():
   getSummaries()


# Run main function
if __name__ == "__main__":
    main()