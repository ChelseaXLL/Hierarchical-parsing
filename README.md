# Hierarchical-parsing
This repo hosts code for hierarchical parsing.

## Why hierarchical parsing

Parsing is really important as it's the 1st step gathering text data.
But instead of sticking with a nature reading (linear) structure, we could 
also try hierarchical parsing. It will split all texts into different chunks based on
titles, so that each text chunks refers to a concept or lies in the same context, which will
be topic-categorized. Next, all text units within the same group, such as sentences, will be ranked based on the title,
and will be sorted according to their alliances with respect to each title. Once get a hierarchical 
results, it will benefit following NLP tasks like 

- knowledge graph construction
- entity linking
- keyword extraction
- text summarization

## How hierarchical parsing and these modules work?

#### General Process:

1. Get original parsing results, which is linear structure based on PDF files. 
2. Extract titles, and get corresponding indices of them in the raw results.
3. Cut raw parsing results into different text chunks
4. Rank text units within each chunk
5. Re-order the results

#### Modules:

- PdfParser.py: Parse original pdfs
- TitleExtractor.py: Extract titles by font's height.
- TitleExtractBySize.py: Extract titles by font's size.
- Rankers.py: Algorithms for sentences ranking
- hierarchical_parser.py: Generate hierarchical parsing results








