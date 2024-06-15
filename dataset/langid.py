from py3langid.langid import LanguageIdentifier, MODEL_FILE
import json
import os


class LangID():
    def __init__(self):
        # Initialize language identifier with pre-trained model
        self.identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE,  norm_probs=True)

    def __from_json_load_field(self, filepath, key):
        """
        Helper method to read a specific field in a JSON file.

        Args:
            filepath (str): The path to the JSON file
            key (str): The key to extract text from in the JSON file

        Returns:
            obj: the object from the specified key in the JSON file (can be str or obj)
        """

        # Check if file exists
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} not found")

        # Read file and get text from specified key
        with open(filepath) as f:
            json_data = json.load(f)
            text = json_data[key]

        return text

    def identify(self, text):
        """
        Identify language from text

        Args:
            text (str): The text to identify the language of

        Returns:
            (str, float): The identified language (e.g. 'en') and the probability (e.g. 0.9)
        """
        # Use the identifier to classify the language of the text
        return self.identifier.classify(text.lower())


    def identify_from_transcript(self, filepath):
        """
        Identify language from transcript.
        
        Args:
            filepath (str): The path to the json file containing transcripts

        Returns:
            (str, float): The identified language (e.g. 'en') and the probability (e.g. 0.9)
        """
        data = self.__from_json_load_field(filepath, 'transcript')

        # Concatenate all lines in the transcript
        text = "".join([data[index]['text'] for index in range(len(data))])

        # Identify language
        return self.identify(text)

    def identify_from_description(self, filepath):
        """
        Identify language from video description.

        Args:
            filepath (str): The path to the JSON file containing the video description

        Returns:
            (str, float): The identified language (e.g. 'en') and the probability (e.g. 0.9)
        """
        text = self.__from_json_load_field(filepath, 'video_description')
        return self.identify(text)

    def identify_from_title(self, filepath):
        """
        Identify language from video title.

        Args:
            filepath (str): The path to the JSON file containing the video title

        Returns:
            (str, float): The identified language (e.g. 'en') and the probability (e.g. 0.9)
        """
        text = self.__from_json_load_field(filepath, 'video_title')
        return self.identify(text)
