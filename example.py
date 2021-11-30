import os
import pickle

from xml.dom.minidom import parse

from topicModel import TopicModel
from documentSummaries import DocumentSummaries
from bs4 import BeautifulSoup


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


def getFederalDockets():
    dockets = ['APHIS-2006-0044','CPSC-2012-0050', 
               'APHIS-2006-0085', 'APHIS-2009-0017']
    return dockets

def getComments():
    regulations = dict()
    comments = list()
    dockets = getFederalDockets()
    for docket in dockets:
        file_name = 'example_data/' + docket + '.pickle'
        cmts = pickle.load(open(file_name, 'rb'))
        regulations[docket] = cmts
        comments.extend(cmts)
    return regulations, comments


def main(num_topics=15):
    
    regulations, comments = getComments()
    
    topicModel = TopicModel(num_topics)
    topicModel.fit(comments)

    for docket_id, document in regulations.items():
        docSummaries = DocumentSummaries(topicModel, num_dominant_topics=3, number_of_sentences=4)
        docSummaries.summarize(document)
        print(docket_id)
        docSummaries.display()

if __name__ == "__main__":
    main()