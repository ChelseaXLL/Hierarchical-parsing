import warnings
warnings.filterwarnings('ignore')
import re

from typing import List, Dict, Any
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from summarizer import Summarizer
from sentence_transformers import SentenceTransformer, util

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)

# text rank with transformer
class TransformerTextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85, based on the paper
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps, more than enough to stabilize 
        self.text_str = None
        self.sentences = None
        self.pr_vector = None
     
#     def normalize_text(self,text:str):
#         """
#         Det rid of special characters.Translates multiple whitespace into single space character.
#         If there is at least one new line character chunk is replaced
#         by single LF (Unix new line) character.
#         """
#         for s in text:
#             if ord(s) < 5:
#                 text = text.replace(s,'')

                
#         return MULTIPLE_WHITESPACE_PATTERN.sub(self._replace_whitespace, text)
    
#     def _replace_whitespace(self,match):
#         """
#         Replace redundant whitespace with LF character
#         """
#         text = match.group()

#         if "\n" in text or "\r" in text:
#             return "\n"
#         else:
#             return " "


#     def is_blank(self,string):
#         """
#         Returns `True` if string contains only white-space characters
#         or is empty. Otherwise `False` is returned.
#         """
#         return not string or string.isspace()


    def get_symmetric_matrix(self,matrix):
        """
        Get Symmetric matrix
        :param matrix:
        :return: matrix
        """
        return matrix + matrix.T - np.diag(matrix.diagonal())


    def core_cosine_similarity(self,vector1, vector2):
        """
        measure cosine similarity between two vectors
        :param vector1:
        :param vector2:
        :return: 0 < cosine similarity value < 1
        """
        return 1 - cosine_distance(vector1, vector2)
    
    # 
    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        model = SentenceTransformer('stsb-roberta-large')
        embedding1 = model.encode(sent1, convert_to_tensor=True)
        embedding2 = model.encode(sent1, convert_to_tensor=True)
        cosine_scores = np.float(util.pytorch_cos_sim(embedding1, embedding2))
        return cosine_scores

    def _build_similarity_matrix(self, sentences):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2])

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)

        # Ignore the 0 element in norm
        sm_norm = np.divide(sm, norm, where=norm != 0)  

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def rank_text(self):
        
        
        number = len(self.sentences)

        top_sentences = []

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                sent = self.sentences[sorted_pr[index]]
#                 sent = self.normalize_text(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

#         tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(self.sentences)

        self.pr_vector = self._run_page_rank(similarity_matrix)
        

# text rank (common tokens)
class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85, based on the paper
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps, more than enough to stabilize 
        self.text_str = None
        self.sentences = None
        self.pr_vector = None
     
#     def normalize_text(self,text:str):
#         """
#         Det rid of special characters.Translates multiple whitespace into single space character.
#         If there is at least one new line character chunk is replaced
#         by single LF (Unix new line) character.
#         """
#         for s in text:
#             if ord(s) < 5:
#                 text = text.replace(s,'')

                
#         return MULTIPLE_WHITESPACE_PATTERN.sub(self._replace_whitespace, text)
    
#     def _replace_whitespace(self,match):
#         """
#         Replace redundant whitespace with LF character
#         """
#         text = match.group()

#         if "\n" in text or "\r" in text:
#             return "\n"
#         else:
#             return " "


#     def is_blank(self,string):
#         """
#         Returns `True` if string contains only white-space characters
#         or is empty. Otherwise `False` is returned.
#         """
#         return not string or string.isspace()


    def get_symmetric_matrix(self,matrix):
        """
        Get Symmetric matrix
        :param matrix:
        :return: matrix
        """
        return matrix + matrix.T - np.diag(matrix.diagonal())


    def core_cosine_similarity(self,vector1, vector2):
        """
        measure cosine similarity between two vectors
        :param vector1:
        :param vector2:
        :return: 0 < cosine similarity value < 1
        """
        return 1 - cosine_distance(vector1, vector2)
    
    # 
    def _sentence_similarity(self, sent1, sent2, stopwords=None):
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

        return self.core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = self.get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def rank_text(self):
        
        
        number = len(self.sentences)

        top_sentences = []

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                sent = self.sentences[sorted_pr[index]]
#                 sent = self.normalize_text(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)

# bert
class Bert4Sentences():
    def __init__(self):
        self.text_str = None
        self.sentences = None
    
    def rank_text(self):

        number = len(self.sentences)
            
        model = Summarizer('bert-large-uncased')
        
        bert_text = model(self.text_str, num_sentences = number)
        bert_sentences = sent_tokenize(bert_text)
        
        return bert_sentences
        
        
    def analyze(self, text):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)