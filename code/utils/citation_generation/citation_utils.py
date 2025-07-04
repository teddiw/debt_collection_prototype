# utils/citation_utils.py

import re

def find_quote(quote, source):
    # Escape special characters in substring, then replace all whitespace sequences with \s+
    pattern = re.escape(quote).replace(r'\ ', r'\s*') 
    # Replace newline characters with \s* to match any whitespace including newlines
    pattern = pattern.replace('\\\n', r'\s*')
    return re.search(pattern, source, re.IGNORECASE) is not None

def highlight_source(quote, source, replacement):
    # Escape special characters in substring, then replace all whitespace sequences with \s+
    pattern = re.escape(quote).replace(r'\ ', r'\s*') 
    # Replace newline characters with \s* to match any whitespace including newlines
    pattern = pattern.replace('\\\n', r'\s*')
    return re.sub(pattern, replacement, source)

def remove_citations(cited_response):
    """
    Remove citations from the cited response.
    """
    # Remove citations in the format [1], [2], etc.
    cited_response = re.sub(r'\s(?:\[\d+\])+(?=\s*[.!?])', '', cited_response)
    
    return cited_response.strip()