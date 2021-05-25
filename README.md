# Hierarchical-parsing
This repo hosts code for hierarchical parsing.

## Why Heierarchical parsing

Parsing is really important as it's the 1st step gathering text data.
But instead of sticking with a nature reading (linear) structure, we could 
also try hierarchical parsing. It will split all texts into different chunks based on
titles, so that each text chunks refers to a concept or lies in the same context, which will
be topic-categorized. Next, each text units, such as sentences, will be ranked based on the title,
and will be sorted according to their alliances with respect to each title. Once get a hierarchical 
results, it will benefit following NLP tasks like 
- knowledge graph construction
- entity linking
- keyword extraction
- text summarization

