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


def get_font_and_size(block):   
    size_distrib = dict()
    font_distrib = dict()
    for l in block["lines"]:
        for s in l["spans"]:
            txt = s["text"].strip()
            size = round(s["size"], 2)
            font = s["font"]
            if size not in size_distrib:
                size_distrib[size] = len(txt)
            else:
                size_distrib[size] += len(txt)
            if font not in font_distrib:
                font_distrib[font] = len(txt)
            else:
                font_distrib[font] += len(txt)
    font_size = max(size_distrib.keys(), key=lambda u: size_distrib[u])
    font_family = max(font_distrib.keys(), key=lambda u: font_distrib[u])
    return font_size, font_family


def is_toc_block(text: str):
    return bool(re.match(r'.{5,} *\d', text))


def is_list(text: str):
    return bool(re.match(r'^((\d+|[a-z])(.| -)|\*) ', text))


def refine_text_block(text):
    text = text.strip()
    text = re.sub(r'(?<=\n)[\t ]+', '', text)
    text = re.sub(r'[\t ]+', ' ', text)
    text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)
    text = re.sub(r'“|”', '"', text)
    text = re.sub(r'…', '..', text)
    text = re.sub(r'’', '\'', text)
    text = re.sub(r'—', '-', text)
    text = re.sub(r'\**[•●‣⁃⁌⁍∙○◘◦☙❥❧⦾⦿]', '*', text)
    text = re.sub(r'。', '. ', text)
    lines = text.split('\n')
    ref = ""
    for l in lines:
        if is_list(l):
            ref += '\n%s' % l.strip()
        else:
            ref += ' %s' % l.strip()
    return ref.strip()


def split_block(block):
    txt = block['text']
    txt = re.sub(r'\n\t* *', '\n', txt)
    texts = txt.split('\n\n')
    split_blocks = []
    for t in texts:
        bl = block.copy()
        bl['text'] = t
        split_blocks.append(bl)
    return split_blocks


def construct_blocks(pdf, page_number, min_font_size: float = 8, sort_by_reading_order=False):
    raw_blocks = get_raw_blocks(pdf, page_number)
    blocks = []
    size_distrib = dict()
    for rb in raw_blocks:
        s = rb['text'].strip()
        size, font = get_font_and_size(rb)
        if size < min_font_size:
            continue
        blocks.append({ "text": s, "size": size, "font": font, "coords": rb["coordinates"] })
        if size not in size_distrib:
            size_distrib[size] = len(s)
        else: 
            size_distrib[size] += len(s)
    standard_font_size = max(size_distrib.keys(), key=lambda u: size_distrib[u]) if len(size_distrib) > 0 else -1
    
    for b in blocks:
        b['text'] = refine_text_block(b['text'])
        if b['size'] == standard_font_size:
            b['simple_size'] = 'normal'
        elif b['size'] < standard_font_size: 
            b['simple_size'] = 'small'
        else:
            b['simple_size'] = 'big'

    if sort_by_reading_order:
        blocks = sorted(blocks, key=lambda u: (u["coords"][3], u["coords"][0]))
    
    new_blocks = []

    for b in blocks:
        if is_toc_block(b['text']):
            continue
        nbs = split_block(b)
        new_blocks += nbs
    return new_blocks


def to_combine(text1, text2):
    if text1.endswith(PUNCTS):
        if is_list(text2):
            return True
        return False
    return True


def merge_into_paragraphs(blocks):
    paras = []
    i = 0
    while i < len(blocks):
        para = blocks[i].copy()
        j = i+1
        while j < len(blocks):
            nxt_block = blocks[j]
            if (not to_combine(para['text'], nxt_block['text'])) \
                or nxt_block['size'] != para['size'] \
                or nxt_block['font'][:6] != para['font'][:6]:
                break
            para['text'] += '\n%s' % nxt_block['text']
            j += 1
        para['text'] = refine_text_block(para['text'])
        
        if para['simple_size'] == 'big':
            para['rep'] = '### %s' % para['text']
        else: 
            para['rep'] = para['text']
        paras.append(para)
        i = j
    return paras


def extract_text_page(
    paragraphs: List[dict],
    sentencizer: object,
    max_words: int=128,
    min_words: int=3
) -> List[Dict]:
    """Re-format the page text and output all phrases.
    Arguments:
        paragraphs {List[str]} -- list of paragraphs in the page
    Keyword Arguments:
        max_words {int} -- maximum number of words in a phrase (default {64})
        min_words {int} -- minimum number of words in a phrase (default {3})
        lang {str} -- document dominant language (default {'en'})
    Returns:
        List[Dict] -- return a list of phrases in the page.
    """
    phrases = []
    for i, p in enumerate(paragraphs):
        sents = sentencizer.segment(p['text'])
        for s in sents:
            s = str(s)
            if min_words <= len(s.split()) <= max_words:
                phrases.append({ 'text': s.strip(), 'paragraph': i+1, 'page': p["page"] })
    return phrases


def create_text_chunks(
    paragraphs: List[Dict],
    phrases: List[Dict], 
    min_words: int=16, 
    max_words: int=64
) -> List[Dict[str, Any]]:
    """Create text chunks from the list of all phrases in a pdf page
    Arguments:
        paragraphs {List[Dict]} -- List of paragraphs in the page
        phrases {List[str]} -- List of phrases in the page
        page_number {int} -- page number
    Keyword Arguments:
        min_words {int} -- minimum number of words in a phrase (default: {16})
        max_words {int} -- maximum number of words in a phrase (default: {128})
    Returns:
        List[Dict[str, Any]] -- List of text chunks
    """
    chunks = []
    for i, p in enumerate(phrases):
        text = p['text']
        page = p['page']
        current_para = p['paragraph']-1
        j = i+1
        while j < len(phrases):
            if len(text.split()) > min_words:
                paras = '\n'.join([para['rep'].strip() for para in paragraphs[p['paragraph']-1: current_para+1]])
                chunks.append({
                    'text': text.strip(),
                    'meta': {
                        'paragraphs': paras,
                        'paragraph_range': '%d-%d' % (p['paragraph']-1, current_para),
                        'page': page
                    }
                })
                
            para_index = phrases[j]['paragraph']-1
            if para_index > current_para:
                if paragraphs[para_index]['simple_size'] != paragraphs[current_para]['simple_size'] \
                or paragraphs[para_index]['font'] != paragraphs[current_para]['font']:
                    break
                current_para = para_index
            
            text_ = text + ' ' + phrases[j]['text']

            if len(text_.split()) > max_words:
                break
            
            text = text_
            j += 1
    logging.info('number of chunks: %d' % len(chunks))
    return chunks
    

def simple_form(text, nlp):
    doc = nlp(text)
    txt = ''
    for tk in doc:
        if tk.like_num or \
            re.match(r'^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$', tk.text): # roman numerals match
            txt += "<num>"
        else:
            txt += tk.text
        txt += tk.whitespace_
    return txt

def parse_pdf_paras(
    filepath: str, 
    first_page: int=0, 
    last_page: int=None,
    min_words: int=16,
    max_words: int=64,
    lang: str='en',
    repetition_threshold: float = 0.7,
    sort_by_reading_order: bool = False
) -> List[Dict[str, Any]]:
    """parse a pdf to a list of dict objects
    Arguments:
        filename {str} -- relative or absolute path to the pdf file
    Keyword Arguments:
        first_page {int} -- first page to be parsed (default: {0})
        last_page {int} -- last page to be parsed (default: {None})
        min_words {int} -- minimum number of words in a phrase
        max_words {int} -- maximum number of words in a phrase
        (Number of words in each phrase must be between `min_words` and `max_words`)
        lang {str} -- sentencizer language (default('en'))
    Returns:
        List[Dict[str, Any]] -- A list of dicts that store all text pieces extracted from the pdf file
    """
    assert lang in ('fr', 'en'), "only 'fr' and 'en' are supported!"

    # open book
    pdf = fitz.open(filepath)
    fn = os.path.basename(filepath)

    if pdf.pageCount < 3:
        repetition_threshold = 1

    # create sentencizer
    logging.info("Creating sentence tokenizer...")
    seg = pysbd.Segmenter(language=lang, clean=False)

#     init spacy model
    logging.info("Init spacy model...")
    if lang == 'en':
        nlp = spacy.load("en_core_web_lg")
    else:
        nlp = spacy.load("fr_core_news_lg")

    # extract phrases
    logging.info('Processing file "%s" - Number of pages: %d.' % (fn, pdf.pageCount))

    # if last_page is None, last_page will be set to the last page
    if last_page is None:
        last_page = pdf.pageCount-1

    paragraphs = []
    chunks = []
    texts = {}
        
    logging.info("Get the blocks...")
    pdf_blocks = {}
    for i, page in enumerate(pdf):
        if i < first_page or i > last_page:
            continue
        logging.info('Processing page: %d' % (i+1))

        blocks = construct_blocks(pdf, i, sort_by_reading_order=sort_by_reading_order)
        pdf_blocks[i] = blocks
        for j, b in enumerate(blocks):    
            s = simple_form(b["text"], nlp)
            if s not in texts:
                texts[s] = [(i, j)]
            else:
                texts[s].append((i, j))

    repeated_text = filter(
        lambda u: len(u[1]) > pdf.pageCount * repetition_threshold,
        sorted([(k, v) for k, v in texts.items()], key=lambda u: len(u[1]), reverse=True),
    )

    to_delete = {}
    for _, ls in repeated_text:
        for page, block_idx in ls:
            if page in to_delete:
                to_delete[page].append(block_idx)
            else:
                to_delete[page] = [block_idx]

    for i in pdf_blocks.keys():
        if i in to_delete:
            nw_blocks = []
            for j, b in enumerate(pdf_blocks[i]):
                if j in to_delete[i]:
                    continue 
                nw_blocks.append(b)
            pdf_blocks[i] = nw_blocks
    
    logging.info("Merge blocks into paragraphs...")
    all_blocks = []
    for i, page in enumerate(pdf):
        if i < first_page or i > last_page:
            continue
        for b in pdf_blocks[i]:
            b["page"] = i+1
            b["filename"] = fn
            all_blocks.append(b)

    paras = merge_into_paragraphs(all_blocks)
    phrases = extract_text_page(paras, seg)
    chunks = create_text_chunks(paras, phrases, min_words=min_words, max_words=max_words)
    for i in range(len(chunks)):
        chunks[i]['meta']['filename'] = fn
    
    return paras


def parse_pdf_chunks(
    filepath: str, 
    first_page: int=0, 
    last_page: int=None,
    min_words: int=16,
    max_words: int=64,
    lang: str='en',
    repetition_threshold: float = 0.7,
    sort_by_reading_order: bool = False
) -> List[Dict[str, Any]]:
    """parse a pdf to a list of dict objects
    Arguments:
        filename {str} -- relative or absolute path to the pdf file
    Keyword Arguments:
        first_page {int} -- first page to be parsed (default: {0})
        last_page {int} -- last page to be parsed (default: {None})
        min_words {int} -- minimum number of words in a phrase
        max_words {int} -- maximum number of words in a phrase
        (Number of words in each phrase must be between `min_words` and `max_words`)
        lang {str} -- sentencizer language (default('en'))
    Returns:
        List[Dict[str, Any]] -- A list of dicts that store all text pieces extracted from the pdf file
    """
    assert lang in ('fr', 'en'), "only 'fr' and 'en' are supported!"

    # open book
    pdf = fitz.open(filepath)
    fn = os.path.basename(filepath)

    if pdf.pageCount < 3:
        repetition_threshold = 1

    # create sentencizer
    logging.info("Creating sentence tokenizer...")
    seg = pysbd.Segmenter(language=lang, clean=False)

#     init spacy model
    logging.info("Init spacy model...")
    if lang == 'en':
        nlp = spacy.load("en_core_web_lg")
    else:
        nlp = spacy.load("fr_core_news_lg")

    # extract phrases
    logging.info('Processing file "%s" - Number of pages: %d.' % (fn, pdf.pageCount))

    # if last_page is None, last_page will be set to the last page
    if last_page is None:
        last_page = pdf.pageCount-1

    paragraphs = []
    chunks = []
    texts = {}
        
    logging.info("Get the blocks...")
    pdf_blocks = {}
    for i, page in enumerate(pdf):
        if i < first_page or i > last_page:
            continue
        logging.info('Processing page: %d' % (i+1))

        blocks = construct_blocks(pdf, i, sort_by_reading_order=sort_by_reading_order)
        pdf_blocks[i] = blocks
        for j, b in enumerate(blocks):    
            s = simple_form(b["text"], nlp)
            if s not in texts:
                texts[s] = [(i, j)]
            else:
                texts[s].append((i, j))

    repeated_text = filter(
        lambda u: len(u[1]) > pdf.pageCount * repetition_threshold,
        sorted([(k, v) for k, v in texts.items()], key=lambda u: len(u[1]), reverse=True),
    )

    to_delete = {}
    for _, ls in repeated_text:
        for page, block_idx in ls:
            if page in to_delete:
                to_delete[page].append(block_idx)
            else:
                to_delete[page] = [block_idx]

    for i in pdf_blocks.keys():
        if i in to_delete:
            nw_blocks = []
            for j, b in enumerate(pdf_blocks[i]):
                if j in to_delete[i]:
                    continue 
                nw_blocks.append(b)
            pdf_blocks[i] = nw_blocks
    
    logging.info("Merge blocks into paragraphs...")
    all_blocks = []
    for i, page in enumerate(pdf):
        if i < first_page or i > last_page:
            continue
        for b in pdf_blocks[i]:
            b["page"] = i+1
            b["filename"] = fn
            all_blocks.append(b)

    paras = merge_into_paragraphs(all_blocks)
    phrases = extract_text_page(paras, seg)
    chunks = create_text_chunks(paras, phrases, min_words=min_words, max_words=max_words)
    for i in range(len(chunks)):
        chunks[i]['meta']['filename'] = fn
    
    return chunks