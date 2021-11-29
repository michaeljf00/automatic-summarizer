import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from ntlk.corpus import stopwords
from ntlk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import coherencemodel
import matplotlib.pylost as plt

from datasets import load_dataset
dataset = load_dataset("cnn_dailymail")



# TODO:
# Create a summarizer that generates new text of the whole document
# where there different areas of information has a smooth transition between
# each one. 

# EXTRA: 
# Use topic-based summary in passage-based query expansion


# Topical summarization: Approaches involve two steps: 
# 1. Identifying topics for the document. 
# 2. Identifying topic-relevant sentences and assigning higher weightage to them to be considered for summarization.

class Summarizer:

	def __init__(self):
		self.summary_length = 200
		

	# Automatically produces a topic-based summary of a text document
	def summarize():
		# Input: document, specified length
		# Output: topic-based summary of document 
		pass


def main():
	pass

if __name__ == "__main__":
	main()