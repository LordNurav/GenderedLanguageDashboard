import spacy
import pandas as pd
import nltk
nltk.data.path.append('C:\\Users\\newbo\\nltk_data')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag, word_tokenize
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
custom_stopwords = {"s", "much", "little", "last", "speak", "say", "second"}  # Add custom stopwords here
stop_words.update(custom_stopwords)

def process_text(filepath):
    nlp.max_length = 2500000  # Set to a value larger than the max text size, in this case Moby Dick
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def extract_descriptors(sentences, character):
    descriptors = []
    for sentence in sentences:
        if character in sentence:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            descriptors.extend([word.lower() for word, tag in tagged if tag in ['JJ', 'VB'] and word.lower() not in stop_words])
    return descriptors

def analyze_sentiment(descriptors):
    return [TextBlob(desc).sentiment.polarity for desc in descriptors]

def preprocess(files, character_metadata):
    results = []
    for filepath in files:
        book_title = filepath.split('/')[-1].split('.')[0]  # Extract book title from file path
        sentences = process_text(filepath)
        for _, row in character_metadata.iterrows():
            descriptors = extract_descriptors(sentences, row['character'])
            sentiment = analyze_sentiment(descriptors)
            for i in range(len(descriptors)):
                results.append({
                    'book': book_title,
                    'character': row['character'],
                    'gender': row['gender'],
                    'descriptor': descriptors[i],
                    'sentiment': sentiment[i]
                })
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    character_metadata = pd.read_csv("character_metadata.csv")
    files = ['./texts/pride_and_prejudice.txt', './texts/moby_dick.txt', './texts/jane_eyre.txt', './texts/anne_of_green_gables.txt', './texts/the_scarlet_pimpernel.txt']  # List of text files to process
    data = preprocess(files, character_metadata)
    data.to_csv("results.csv", index=False)
    print("Preprocessing complete!")
