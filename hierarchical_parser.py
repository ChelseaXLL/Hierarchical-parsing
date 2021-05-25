from typing import List, Dict, Any

import fitz
from collections import defaultdict

import string

import Rankers
import TitleExtractor
import PdfParser
import json


import os
import re
from itertools import groupby


def get_text_chunks(pdf: str):
    '''
    Parse the pdf file to return a list of texts  
    '''
    paras = PdfParser.parse_pdf_paras(pdf)
    chunks = [item['text'] for item in paras]
    
    return chunks

def extract_titles(pdf: str, 
                   main_title_value: float, 
                   subtitle_value: float, 
                   start: int = None, 
                   end: int = None, 
                   stop: int = None):
    
    '''Extract all main titles and sutitles from pdf, based on given values of characters'height
       Start: start page (If none, then starts with 1st page)
       End: last page (If none, then ends with last page)
       Stop: interval (1 by default)
    '''
    # open pdf files
    pdf_ = fitz.open(pdf)
    
    if start == None:
        start = 0
    
    if end == None:
        end = len(pdf_)
        
    if stop == None:
        stop = 1
    
    # main titles
    titles_dict = TitleExtractor.get_titles(pdf_, start, end, stop, main_title_value)
    
    # subtitles
    subtitles_dict = TitleExtractor.get_titles(pdf_, start, end, stop, subtitle_value)
    
    # merge titles
    titles_list = list(titles_dict.values())
    subtitles_list = list(subtitles_dict.values())
    all_titles = list(set(titles_list + subtitles_list))
    
    
    return all_titles


def get_title_index(text_chunks: List[str], titles: List[str]):
    '''
    This function will
     1. Get all titles in the original parsing results
     2. Get indices of all titles in the parsing results
    '''
    titles_in = []
    for t in titles:
        if t in text_chunks:
            titles_in.append(t)
            
    title_index = []
    for title in titles_in:
        title_index.append(text_chunks.index(title))
    title_index_ = sorted(title_index)
            
    return title_index_


def get_title_text(text_chunks: List[str], title_index: List[int]):
    '''
    This function will group parsing results into different
    text chunks based on corresponding title.
    It will output a list of sublist containing different text units,
    and the 1st item of each text units is the title itself
    '''
    text_units = []
    text_units.append(text_chunks[:title_index[0]])
    
    for i in range(1,len(title_index)):
         text_units.append(text_chunks[title_index[i-1]:title_index[i]])
            
    text_units.append(text_chunks[title_index[-1]:])
    
    for t in text_units:
        if len(t) ==0:
            text_units.remove(t)
            
    return text_units


def run_transfomer_textrank(text_chunk: List[str]):
    '''
    Run textrank algorithm with bert transformer
    '''
    ranked_text = []
    textrank = Rankers.TransformerTextRank4Sentences()
    textrank.analyze(''.join(text_chunk[1:]))
    tr_results = textrank.rank_text()
    ranked_text.append(text_chunk[0])
    ranked_text.extend(tr_results)
    
    return ranked_text

def run_textrank(text_chunk: List[str]):
    '''
    Run original textrank algorithm
    '''
    ranked_text = []
    textrank = Rankers.TextRank4Sentences()
    textrank.analyze(''.join(text_chunk[1:]))
    tr_results = textrank.rank_text()
    ranked_text.append(text_chunk[0])
    ranked_text.extend(tr_results)
    
    return ranked_text



def rank_text(text_units: List[str], all_titles: List[str]):
    
    '''
    Initialize textrank to get ranked sentences based on each title
    '''
    texts = None
    ranked_results = []
    if text_units[0][0] in all_titles:
        texts = text_units[:]
        for s in texts:
            if len(s) > 1:
                ranked_results.append(run_textrank(s))
        
    else:
        text_no_title = text_units[0]
        texts = text_units[1:]
        for s in texts:
            if len(s) > 1:
                ranked_results.append(run_textrank(s))
        ranked_results.insert(0, text_no_title)
        
    return ranked_results


def rank_text_transformer(text_units, all_titles):
    '''
    Runing textrank with transformers, which will take lots of time, better use for final ouput run
    '''
    texts = None
    ranked_results = []
    if text_units[0][0] in all_titles:
        texts = text_units[:]
        for s in texts:
            if len(s) > 1:
                ranked_results.append(run_transfomer_textrank(s))
        
    else:
        text_no_title = text_units[0]
        texts = text_units[1:]
        for s in texts:
            if len(s) > 1:
                ranked_results.append(run_transfomer_textrank(s))
        ranked_results.insert(0, text_no_title)
        
    return ranked_results           

def hierarchical_parser(pdf: str, 
                        main_title_value: float, 
                        subtitle_value: float, 
                        start: int = None, 
                        end: int = None, 
                        stop: int = None):
    '''
    Parse the pdf, and get hierarchical result,
    which is a list of sublists of text units. 
    **Still under improvement**
    '''
    
    text_chunks = get_text_chunks(pdf)
    
    all_titles = extract_titles(pdf, main_title_value, subtitle_value)
    
    title_index = get_title_index(text_chunks, all_titles)
    
    text_units = get_title_text(text_chunks, title_index)
    
    ranked_results = rank_text(text_units, all_titles)
    
    return ranked_results