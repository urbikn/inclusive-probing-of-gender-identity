import re
import spacy

class Preprocessing:
    # we remove text chunks which do not correspond to the actual spoken text,
    # like the information of the speaker name (Speaker 1: ...) annotations ([laughs], \laughs\, (laughs))
    # and other non-linguistic information (e.g., [music]).
    regex_pattern_annotations = re.compile(r'\[.*?\]|\(.*?\)|\\.*?\\')
    regex_pattern_speaker_name = re.compile(r'\s\w+:|\s\w+\s\d:')
    
    # we discard captions if they contain any character that is not an English letter, apostrophe, comma, dot, and white space,
    # regex_pattern_non_english_letters = re.compile(r'[\-]')
    regex_patter_non_ascii = re.compile(r'[^\x00-\x7f]')

    # python -m spacy download en_core_web_sm
    spacy_en = spacy.load('en_core_web_sm')

    def __init__(self) -> None:
        pass

    @staticmethod
    def __remove_less_than_one_second(transcript):
        """ Remove the words that are less than one second and more than 10 seconds

        Args:
            transcript (list): List of dictionary of words and their duration
        
        Returns:
            transcript (list): original transcript with words that are less than one second and more than 10 seconds removed
        """
        for item in transcript:
            if item['duration'] < 1 or item['duration'] > 10:
                transcript.remove(item)
        
        return transcript
    
    @staticmethod
    def __remove_url(text):
        """ Remove the url from the text (when it starts with 'http' or 'www')"""
        return " ".join([word for word in text.split() if (not word.startswith('http') or not word.startswith('www'))])

    @staticmethod
    def remove_speaker_name_and_annotation(text):
        """ Remove the speaker name and annotation from the text, e.g., Speaker 1: ... [laughs] \laughs\ (laughs)
        """
        text = Preprocessing.regex_pattern_annotations.sub('', text)
        text = Preprocessing.regex_pattern_speaker_name.sub('', text)

        return text

    @staticmethod
    def remove_non_english_letters(text):
        """ Keep only ASCII characters
        """
        text = text.replace('’', '\'') # replace the dash with space

        return Preprocessing.regex_patter_non_ascii.sub('', text)

    @staticmethod
    def filter(transcript):
        # First remove the words that are less than one second and more than 10 seconds
        transcript = Preprocessing.__remove_less_than_one_second(transcript)

        for i, item in enumerate(transcript):
            text = item['text']
            
            # Remove the url from the text (when it starts with 'http' or 'www')
            text = Preprocessing.__remove_url(text)

            # Remove the speaker name and annotation from the text, e.g., Speaker 1: ... [laughs] \laughs\ (laughs)
            text = Preprocessing.remove_speaker_name_and_annotation(text)

            # Remove the non-english letters
            text = Preprocessing.remove_non_english_letters(text)

            transcript[i]['text'] = text

        return transcript
    
    @staticmethod
    def to_sentences(text):
        """ Split the text into sentences using spacy
        """
        sentences = [str(sent) for sent in Preprocessing.spacy_en(text).sents]

        return sentences