import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load necessary resources

nlp = spacy.load("en_core_web_sm")


# Preprocess text to lowercase
def preprocess_text(text):
    lower_words = text.lower()
    return lower_words


# Add spacing around punctuation
def add_spacing_around_punctuation(text):
    tokens = word_tokenize(text)
    corrected_text = " ".join([word if word.isalpha() else f" {word} " for word in tokens])
    return corrected_text


# Identify missing punctuation using spaCy
def identify_missing_punctuation(original_text, transcription):
    nlp = spacy.load("en_core_web_sm")

    original_doc = nlp(original_text)
    transcription_doc = nlp(transcription)

    original_punctuation = [token.text for token in original_doc if token.is_punct]
    transcription_punctuation = [token.text for token in transcription_doc if token.is_punct]

    valid_punctuation = ['?', '!', ',', '.']

    missing_punctuation = [p for p in valid_punctuation if
                           p in original_punctuation and p not in transcription_punctuation]

    return missing_punctuation


# Identify missing words using gensim
def identify_missing_words(original_words, transcription_words):
    missing_words = [word for word in original_words if word not in transcription_words]
    return missing_words


# Identify incorrect words using NLTK
def identify_incorrect_words(original_words, transcription_words):
    incorrect_words = [word for word in transcription_words if
                       word not in original_words and word not in ('.', ',', '!', '?')]
    return incorrect_words

    # Identify incorrect punctuation using spaCy


def identify_incorrect_punctuation(original_text, transcription):
    nlp = spacy.load("en_core_web_sm")

    original_doc = nlp(original_text)
    transcription_doc = nlp(transcription)

    original_punctuation = [token.text for token in original_doc if token.is_punct]
    transcription_punctuation = [token.text for token in transcription_doc if token.is_punct]

    incorrect_punctuation = [p for p in transcription_punctuation if p not in original_punctuation]

    return incorrect_punctuation


def tokenize_sentences(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    return sentences


def tokenize_words(sentence):
    # Tokenize sentence into words
    words = word_tokenize(sentence)
    return words

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    original_text = data['original_text']
    transcription = data['transcription']

    original_words = preprocess_text(original_text).split()
    transcription_words = preprocess_text(transcription).split()

    corrected_text_original = add_spacing_around_punctuation(original_text)
    corrected_text_transcription = add_spacing_around_punctuation(transcription)

    original_words = preprocess_text(corrected_text_original).split()
    transcription_words = preprocess_text(corrected_text_transcription).split()

    missing_punctuation = identify_missing_punctuation(corrected_text_original, corrected_text_transcription)
    missing_words = identify_missing_words(original_words, transcription_words)
    incorrect_words = identify_incorrect_words(original_words, transcription_words)
    incorrect_punctuation = identify_incorrect_punctuation(corrected_text_original, corrected_text_transcription)

    original_sentences = tokenize_sentences(corrected_text_original)
    transcription_sentences = tokenize_sentences(corrected_text_transcription)

    original_words_list = [tokenize_words(sentence) for sentence in original_sentences]
    transcription_words_list = [tokenize_words(sentence) for sentence in transcription_sentences]

    missing_words = []
    for original_words, transcription_words in zip(original_words_list, transcription_words_list):
        missing = [word for word in original_words if
                   word not in transcription_words and word.lower() not in [w.lower() for w in transcription_words]]
        missing_words.extend(missing)

    missing_words = [word for word in missing_words if word not in missing_punctuation]

    result = {
        "missing_words": missing_words,
        "missing_punctuation": missing_punctuation,
        "incorrect_words": incorrect_words,
        "incorrect_punctuation": incorrect_punctuation
    }

    return jsonify(result)

