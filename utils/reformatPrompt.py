import warnings
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup
import unicodedata
import re

def remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without HTML-Tags
    '''
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without URL addresses
    '''
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_accented_chars_func(text):
    '''
    Removes all accented characters from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)

def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string without extra whitespaces
    '''
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

def word_count_func(text):
    '''
    Counts words within a string

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Number of words within a string, integer
    '''
    return len(text.split())

def reformatPrompt(prompt):
  prompt = prompt.lower()
  prompt = remove_html_tags_func(prompt)
  prompt = remove_url_func(prompt)
  prompt = remove_accented_chars_func(prompt)
  prompt = remove_punctuation_func(prompt)
  prompt = remove_irr_char_func(prompt)
  prompt = remove_extra_whitespaces_func(prompt)
  return prompt