from SPARQLWrapper import SPARQLWrapper, JSON
import os
import time
import uuid
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class WikiDataAPI:
    def __init__(self):
        self.sparql = SPARQLWrapper(
            "https://query.wikidata.org/sparql",
            agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11"
        )
        self.sparql.setReturnFormat(JSON)

        self.label_to_id = {
            'non-binary': 'wd:Q48270',
            'trans woman': 'wd:Q1052281',
            'trans man': 'wd:Q2449503',
            'cis woman': 'wd:Q6581072',
            'trans man': 'wd:Q6581097',
        }

    def get_youtubers_api(self, gender_identities=['trans woman', 'trans man'], limit=0, offset=0):
        """
        Retrieves YouTubers from Wikidata that have a specified gender identity.

        Args:
            gender_identities (list of str): List of gender identities to filter YouTubers by.
                Each gender identity should be a string that matches a label in `self.label_to_id`.

        Returns:
            pandas.DataFrame: A DataFrame containing information about the retrieved YouTubers.
            The DataFrame has the following columns:
                - entity: Wikidata ID of the YouTuber
                - name: Name of the YouTuber (if available)
                - genderLabel: Gender identity of the YouTuber (as a Wikidata ID)
                - youtube_channel: URL of the YouTuber's YouTube channel
        """

        # First convert gender identity to the corresponding wikidata ID
        gender_ids = [self.label_to_id[gender_identity] for gender_identity in gender_identities]

        # Run the query depending on the number of gender identities provided
        if len(gender_ids) > 1:
            self.sparql.setQuery("""
                SELECT ?entity ?name ?genderLabel (CONCAT("https://www.youtube.com/channel/", ?youtube_id) AS ?youtube_channel)
                WHERE {
                    ?entity wdt:P31 wd:Q5 . # Instance of a human
                    ?entity wdt:P21 ?gender . # Gather their gender identity label
                    FILTER ( ?gender in ( """ +
                    ",".join(gender_ids) + # Only select the following gender identities
                """  ) )

                    # Filtering Country
                    ?entity wdt:P27 ?country
                    FILTER( ?country in (wd:Q30, wd:Q145)) #USA, UK

                    ?entity wdt:P2397 ?youtube_id . # that has a youtube channel ID

                    OPTIONAL {
                        ?entity rdfs:label ?name.
                        FILTER (lang(?name) = "en")
                    }
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } # Add labels to all entries
                } ORDER BY ?name """
            ) 
        else:
            self.sparql.setQuery("""
                SELECT ?entity ?name ?genderLabel (CONCAT("https://www.youtube.com/channel/", ?youtube_id) AS ?youtube_channel)
                WHERE {
                    ?entity wdt:P31 wd:Q5 . # Instance of a human
                    ?entity wdt:P21 ?gender . # Gather their gender identity
                    ?entity wdt:P21 """ + gender_ids[0] + """ .

                    # Filtering Country
                    ?entity wdt:P27 ?country
                    FILTER( ?country in (wd:Q30, wd:Q145)) #USA, UK

                    ?entity wdt:P2397 ?youtube_id . # that has a youtube channel ID
                    OPTIONAL {
                        ?entity rdfs:label ?name.
                        FILTER (lang(?name) = "en")
                    }
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } # Add labels to all entries
                }
                """
            ) 

        query_results = self.sparql.query().convert()
        results = pd.json_normalize(query_results["results"]["bindings"])

        # As post-filtering step, remove all columns with .type or .xml:lang in name
        results = results[results.columns.drop(list(results.filter(regex='\.type')) + list(results.filter(regex='\.xml:lang')))]

        return results
    
    def get_properties_api(self, entry_id):
        """
        Retrieves all properties of a given Wikidata entity ID.

        Args:
            entry_id (str): The Wikidata entity ID to retrieve properties for. This can be any string with the format "Q:12515".

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved properties of the given Wikidata entity ID.
        """
        self.sparql.setQuery("""
            SELECT ?wdLabel ?ps_Label ?wdpqLabel ?pq_Label ?dateCreated ?dateModified {
                VALUES (?person) {(wd:""" + entry_id + """)}
                
                ?person ?property ?statement .
                ?statement ?ps ?ps_ .
    
                ?wd wikibase:claim ?p.
                ?wd wikibase:statementProperty ?ps.
                
                OPTIONAL {
                    ?statement ?pq ?pq_ .
                    ?wdpq wikibase:qualifier ?pq .
                }
    
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
                } ORDER BY ?wd ?statement ?ps_
        """
        ) 

        query_results = self.sparql.query().convert()
        results = pd.json_normalize(query_results["results"]["bindings"])

        # As post-filtering step, remove all columns with .type or .xml:lang in name
        results = results[results.columns.drop(list(results.filter(regex='\.type')) + list(results.filter(regex='\.xml:lang')))]

        return results


class WikiDataUsers(WikiDataAPI):
    def __init__(self) -> None:
        super().__init__()

    def youtubers_to_json(self, gender_identities=['cis woman', 'cis male', 'transfemale', 'transmale', 'non-binary'], save_to='users'):
        """
        Gathers youtubers based on the specified gender identities and saves the result to a JSON file.

        Args:
            gender_identities (list, optional): List of gender identities to filter the youtubers. Defaults to ['cisfemale', 'cismale', 'transfemale', 'transmale', 'non-binary'].
            save_to (str, optional): Path to save the JSON file. Defaults to '.'.

        Returns:
            str: Path to the saved JSON file.
            ids: List of Wikidata IDs of the retrieved youtubers.
        """
        if not os.path.exists(save_to):
            os.mkdir(save_to)

        youtubers = self.get_youtubers_api(gender_identities)
        youtubers.sort_values(by=['name.value'], inplace=True)

        # Get all Wikidata IDs
        user_ids = [user_id.split('/')[-1] for user_id in youtubers['entity.value']]

        # Create a unique ID for each file
        unique_id = str(uuid.uuid4())[:8]
        file_to_save_to = f'{save_to}/original_youtubers-{unique_id}.json'

        # Save to JSON file
        youtubers.to_json(file_to_save_to, orient='records', indent=4)

        return file_to_save_to, user_ids
    
    def all_properties_user(self, user_id):
        """ 
        Retrieves all properties of a given Wikidata user ID.

        Args:
            user_id (str): The Wikidata user ID to retrieve properties for. This can be any string with the format "Q:12515".

        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved properties of the given Wikidata user ID.
        """
        return self.get_properties_api(user_id)
    
    def all_properties_of_users(self, user_ids, save_to='users', timeout=3):
        """
        Retrieves all properties of a given list of Wikidata user IDs and saves the results to JSON files.

        Args:
            user_ids (list of str): List of Wikidata user IDs to retrieve properties for. Each user ID should be a string that matches the format "Q:12515".
            save_to (str, optional): Path to save the JSON files. Defaults to '.'.
            timeout (int, optional): Timeout between each request. Defaults to 3.

        Returns:
            saved_files (list of str): List of paths to the saved JSON files.
        """
        if not os.path.exists(save_to):
            os.mkdir(save_to)

        saved_files = []

        for user_id in tqdm(user_ids, total=len(user_ids), desc=f"Collecting WikiData properties"):
            # Check if the file already exists
            if os.path.exists(f'{save_to}/{user_id}.json'):
                continue

            # Get all properties of the user
            properties = self.all_properties_user(user_id)

            # Get gender identities
            gender_identities = list(properties[properties['wdLabel.value'] == 'sex or gender']['ps_Label.value'].unique())

            user_data = {
                'gender_identities': gender_identities,
                'metadata': properties.to_dict(orient="records")
            }

            file_to_save_to = f'{save_to}/{user_id}.json'
            with open(file_to_save_to, 'w') as f:
                json.dump(user_data, f, indent=4)

            saved_files.append(file_to_save_to)

            # Sleep for a bit to not overload the API
            time.sleep(timeout)
            
        return saved_files
    
    @staticmethod
    def get_youtube_channel(user_file):
        """
        Read the user file (like Q12345.json) and return the YouTube channel URL.

        Args:
            user_file (str): Path to the user file.

        Returns:
            list (str): YouTube channel URL.

        Raises:
            FileNotFoundError: If the user_file does not exist.
        """
        # Check if the file exists
        if not os.path.exists(user_file):
            raise FileNotFoundError(f"File {user_file} does not exist.")


        with open(user_file, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data['metadata'])

        rows = df[df['wdLabel.value'] == 'YouTube channel ID']
        youtube_channel_ids = list(rows['ps_Label.value'].unique())
        youtube_channels = [f'https://www.youtube.com/channel/{channel_id}' for channel_id in youtube_channel_ids]

        return youtube_channels

    @staticmethod
    def is_from_english_speaking_country(user_file):
        """
        Check if the user is from an English-speaking country based on their file.

        Args:
            user_file (str): The path to the user's file.

        Returns:
            bool: True if the user is from an English-speaking country, False otherwise.

        Raises:
            FileNotFoundError: If the user_file does not exist.
        """

        # Check if the file exists
        if not os.path.exists(user_file):
            raise FileNotFoundError(f"File {user_file} does not exist.")

        list_of_english_speaking_countries = [
            'United Kingdom',
            'United States of America',
        ]

        with open(user_file, 'r') as f:
            data = json.load(f)

            if 'metadata' not in data:
                return False

            df = pd.DataFrame(data['metadata'])

        rows = df[df['wdLabel.value'] == 'country of citizenship']

        list_of_english_speaking_countries = [country.lower() for country in list_of_english_speaking_countries]
        countries = [country.lower() for country in list(rows['ps_Label.value'].unique())]

        # See if any of the countries match
        for country in countries:
            if country in list_of_english_speaking_countries:
                return True
        
        return False




if __name__ == "__main__":
    wikidata = WikiDataUsers()

    gender_identities = ['transfemale', 'transmale', 'non-binary']

    # Get all YouTubers
    json_path, user_ids = wikidata.youtubers_to_json(gender_identities)

    # Get all properties of a given user
    if not os.path.exists('users'):
        os.mkdir('users')

    # Get all properties of all users
    saved_files = wikidata.all_properties_of_users(user_ids, save_to='users')

    # filter non-english speaking users
    filtered_users = 0
    for user_file in tqdm(saved_files, desc="Filtering non-English speaking users"):
        if not wikidata.is_from_english_speaking_country(user_file):
            os.remove(user_file)
            filtered_users += 1
    
    print(f"Filtered {filtered_users} users")
