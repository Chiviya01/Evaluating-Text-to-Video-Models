import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
import re 
import string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# Identify key differences between two sentences using POS tags
import difflib
import json
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict
import torch
import sys
import os
import requests
import random
import pandas as pd

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")

import json
def get_all_captions(json_file):
    '''
    To extract original caption and generated captions from json file for a single video
    '''
    with open(json_file) as f:
        data = json.load(f)
    all_lists = []
    for key in data.keys():
        original_caption = key.split('/')[-1]
        for sublist in data[key].values():
            all_lists.append(sublist[0])
    return original_caption, all_lists


def preprocess_text(text:str):
    '''
    Function to preprocess a given text by removing punctuation, stop words, non-English words, converting numerical values to text, and lowercasing the text. Additionally, it removes certain words related to image and video data sources to reduce noise in the resulting tokens.
    Arguments:
    -----------
    text: str - the text to preprocess
    Returns:
    -----------
    str - the preprocessed text
    '''
    # include Imagen, MetaAI, pixabay and stockphoto to removewater marks
    remove_words = ["photo", "stockphoto", "stock", "background",
                "image", "©stockfoto", "©", "pixabay", 
                'highly' 'detailed', "shutterstock", 
                "video", "videos", "royalty-free", "footage", "resolution", 'MetaAI', 'Imagen']
    
    # Convert numerical values to text
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            text = text.replace(ent.text, str(ent.text))
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove stop words and non-english words
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha and token.lang_ == "en" and token.pos_ not in ['DET', 'CCONJ', 'SCONJ', 'PUNCT', 'SPACE',  'ADJ', 'ADV']]
    tokens = [token for token in tokens if token not in remove_words]
    return ' '.join(tokens)
    
def compute_cosine_similarity(sentence_1:str, sentence_2:str) -> float:
    '''
    This function calculates cosine similarity for two sentences
    '''
    # Preprocess texts
    original_text = preprocess_text(sentence_1)
    generated_text = preprocess_text(sentence_2)

    # Compute similarity score
    vectorizer = CountVectorizer().fit_transform([original_text, generated_text])
    similarity_score = cosine_similarity(vectorizer)[0][1]
    return similarity_score

def compute_bert_similarity(sentence_1:str, sentence_2:str) -> float:
    '''
    This function calculates BERT similarity for two sentences
    '''
    # Preprocess texts
    sentence_1 = preprocess_text(sentence_1)
    sentence_2 = preprocess_text(sentence_2)
    
    # Tokenize input sentences
    encoded_sentence_1 = tokenizer(sentence_1, padding=True, truncation=True, return_tensors='pt')
    encoded_sentence_2 = tokenizer(sentence_2, padding=True, truncation=True, return_tensors='pt')
    
    # Generate sentence embeddings
    with torch.no_grad():
        outputs = model(**encoded_sentence_1)
        embeddings_1 = outputs.last_hidden_state[:, 0, :]
        
        outputs = model(**encoded_sentence_2)
        embeddings_2 = outputs.last_hidden_state[:, 0, :]
    
    # Compute similarity score
    similarity_score = torch.nn.functional.cosine_similarity(embeddings_1, embeddings_2)
    
    return similarity_score.item()

def calculate_similarity(sentence1:str, sentence2:str) -> float:
    '''
    This function computes an overall sentences similarity by combining BERT similarity and cosine similarity
    '''
    cosine_similarity = compute_cosine_similarity(sentence1, sentence2)
    bert_similarity = compute_bert_similarity(sentence1, sentence2)
    # BERT sometimes overperforms so reducing the final score by combining it with cosine similarity
    if cosine_similarity != 0:
        overall_score = 0.25 * cosine_similarity + bert_similarity * 0.75
    else:
        overall_score = bert_similarity * 0.5
    if type(overall_score) == float: 
        return overall_score
    else:
        return overall_score.item()

def compute_weighted_similarity(original_caption:str, captions_list):
    '''
    Computes the weighted similarity between an original caption and a list of captions.
    Args:
        original_caption : str
            The original caption to compare with.
        captions_list : list
            A list of captions to compare with.
    Returns:
        tuple
            A tuple containing the best matching caption, its similarity score with the original caption, 
            and the overall weighted similarity score computed using all captions in the list.
    '''
    captions_dict = {}
    for caption in captions_list:
        if str(caption) in captions_dict.keys():
            captions_dict[str(caption)] += 1
        else:
            captions_dict[str(caption)] = 1
    scores = defaultdict(int)
    max_score = 0
    best_caption = ""
    total_weight = 0
    
    # Compute similarity scores for all captions and sum up the weights
    for caption, count in captions_dict.items():
        score = calculate_similarity(original_caption, caption)
        scores[caption] = score
        total_weight += count
        
        # Update the best caption if its score is higher
        if score > max_score:
            max_score = score
            best_caption = caption
    
    # Compute the overall weighted similarity score
    overall_score = 0
    for caption, score in scores.items():
        weight = captions_dict[caption]
        overall_score += score * weight / total_weight
    
    return best_caption, max_score, overall_score

def extract_non_necessary_words(sentence:str):
    doc = nlp(sentence)
    non_necessary_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_digit and token.pos_ not in ["ADJ", "DET"]:
            non_necessary_words.append(token.lemma_)
    return non_necessary_words

def identify_key_differences(original_text:str, generated_text:str):
    word_diff = {
        'pos_differences': set(),
        'words_only_in_original_text': set(),
        'words_only_in_generated_text': set()
    }
    # Preprocess original and generated text
    original_text = extract_non_necessary_words(original_text)
    original_text = ' '.join(original_text)
    generated_text = extract_non_necessary_words(generated_text)
    generated_text = ' '.join(generated_text)
    # POS differences
    pos_original = set(nltk.pos_tag(original_text.split()))
    pos_generated = set(nltk.pos_tag(generated_text.split()))
    word_diff['pos_differences'] = {tag[1] for tag in pos_original ^ pos_generated}
    # Words only in original text
    words_original = set(original_text.split())
    words_generated = set(generated_text.split())
    word_diff['words_only_in_original_text'] = words_original - words_generated
    # Words only in generated text
    word_diff['words_only_in_generated_text'] = words_generated - words_original
    return word_diff

def find_unique_words(s1, s2):
    word_diff_dict = identify_key_differences(s1, s2)
    words_only_in_original_text = word_diff_dict['words_only_in_original_text']
    words_only_in_generated_text = word_diff_dict['words_only_in_generated_text']
    
    unique_words = []
    for word in words_only_in_original_text:
        similarity_scores = []
        for generated_word in words_only_in_generated_text:
            similarity_scores.append(len(set(word).intersection(generated_word)) / len(set(word).union(generated_word)))
        max_similarity_score = max(similarity_scores)
        if max_similarity_score < 0.5:
            unique_words.append(word)
    return unique_words

def main():
    json_file_path = input("Please enter the path to the json file: ")
    if os.path.exists(json_file_path):
        original_caption, captions = get_all_captions(json_file_path)
    else:
        print("File does not exist.")
    original_caption, captions = get_all_captions(json_file_path)
    best_caption, best_score, avg_score = compute_weighted_similarity(original_caption, captions)
    # add the data to the list
    data = []
    data.append({'video_file': original_caption, 'best_caption': best_caption, 'best_score': best_score, 'avg_score': avg_score})
    df = pd.DataFrame(data)
    df.to_csv('video_text_similarity.csv', index=False)
    print('Text Similarity data is stored in video_text_similarity.csv file.')
if __name__ == "__main__":
    main()