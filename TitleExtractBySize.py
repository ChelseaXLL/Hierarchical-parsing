from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
import os
import re
import logging
from typing import List, Dict, Any

import spacy
import fitz
import pysbd
import time


PUNCTS = (".", "!", "?", "\"")


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_ext_ascii(s):
    return all(ord(c) < 256 for c in s)


def is_valid(block):
    if 'lines' not in block:
        return False
    total_spans = 0
    for l in block['lines']:
        total_spans += len(l['spans'])
    return total_spans > 0



def get_raw_blocks(pdf, page_number):
    text_blocks = [(b[4], b[:4]) for b in pdf[page_number].getTextBlocks() if not re.match(r'^<image:', b[4])]
    blocks = [b for b in pdf[page_number].getText('dict')['blocks'] if is_valid(b)]
    for tb, b in zip(text_blocks, blocks):
        b['text'] = tb[0]
        b['coordinates'] = tb[1]
    return blocks

def get_text_size_loc(block):
    info_block = []
    items = block['lines']
    for item in items:
        spans = item['spans']
        for info in spans:
            text_ = info['text']
            size_ = round(info['size'],2)
            origin_ = info['origin']
            area_ = round(fitz.Rect(info['bbox']).getArea(),2)
            bbox_ = info['bbox']
            font_ = info['font']
            flags_ = info['flags']
            asc_ = round(info['ascender'],2)
            block_ = items.index(item)

            info_block.append([size_, text_, font_])
            
    return info_block


def text_size_all_blocks(blocks):
    info_blocks = []
    for i in range(len(blocks)):
        info_block = get_text_size_loc(blocks[i])
        info_blocks.append([info_block,i])
        
    return info_blocks

def get_block_number(info_blocks):
    final_blocks = []
    for sample in info_blocks:
        for item in sample[0]:
            item.append(sample[1])
        sample.pop()
        final_blocks.extend(sample[0])
        
    return final_blocks
        

def get_all_titles(pdf, font_attr, font_size, start = None, end = None):
    pdf_ = fitz.open(pdf)
    
    if start == None:
        start = 0
        
    if end == None:
        end = len(pdf_)
        
    blocks = []
    for i in range(start, end,1):
        block = get_raw_blocks(pdf_, i)
        blocks.extend(block)
        
    info_blocks = text_size_all_blocks(blocks)
    final_blocks = get_block_number(info_blocks)
    
    titles_words = [item[1:] for item in final_blocks if item[0] == font_size if item[2] == font_attr]
    
    titles = defaultdict(list)
    
    for w in titles_words: 
        titles[w[1], w[-1]].append(w[0])
    for k in titles.keys():
        titles[k] = ' '.join(titles[k])

    titles_ = list(titles.values())
    
    return titles_