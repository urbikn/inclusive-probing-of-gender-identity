import argparse
import json
import logging
import os
import random
import shutil
import timeit
import uuid
from pathlib import Path
from time import sleep
import warnings

import pandas as pd
from tqdm import tqdm

from langid import LangID
from preprocessing import Preprocessing
from speaker_diarization import SpeakerDiarization
from wikidata import WikiDataAPI, WikiDataUsers
from youtube import YouTubeAPI, YouTubeVideo

warnings.filterwarnings("ignore", message=".*set_audio_backend has been deprecated.*")


def main():
    parser = argparse.ArgumentParser(description="Create dataset")
    
    # Define arguments
    parser.add_argument('--users_folder', type=str, default='../users', help='Path to the users folder')
    parser.add_argument('--transcripts_folder', type=str, default='../transcripts', help='Path to the transcripts folder')
    parser.add_argument('--gender_identities', type=str, nargs='+', default=['trans woman', 'trans man', 'non-binary', 'cis woman', 'cis man'], help='List')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use parsed arguments
    users_folder = args.users_folder
    transcripts_folder = args.transcripts_folder
    gender_identities = args.gender_identities

    # 1. Get list of YouTubers from WikiData
    wikidata = WikiDataUsers()
    json_path, user_ids = wikidata.youtubers_to_json(gender_identities)

    # Get all properties of all users
    users_folder = Path(users_folder)
    users_folder.mkdir(parents=True, exist_ok=True)
    user_files = wikidata.all_properties_of_users(user_ids, save_to=str(users_folder), timeout=2)

    user_files = [str(users_folder / Path(user_file).name) for user_file in os.listdir(users_folder)]
    
    breakpoint()

    # 2. Filter out non-English users
    filtered_users = 0
    for user_file in tqdm(user_files, desc="Filtering non-English speaking users"):
        if not wikidata.is_from_english_speaking_country(user_file):
            os.remove(user_file)
            filtered_users += 1
    print(f"Filtered {filtered_users} users")

    # 3. Download all captions from YouTube
    transcripts_folder = Path(transcripts_folder)
    transcripts_folder.mkdir(parents=True, exist_ok=True)
    download_captions(user_folder=users_folder, save_to=str(transcripts_folder))

    # 4. Filter out music and trailer files
    filter_music_and_trailer(transcripts_folder)
    correct_user_and_video_ids(users_folder=users_folder, transcript_folder=transcripts_folder)

    # # 5. Do speaker filtering
    filter_speakers_in_transcripts(users_folder, transcripts_folder)
    filter_empty_transcripts(transcripts_folder)
    correct_user_and_video_ids(users_folder, transcripts_folder)

    # 6. Using langid to filter out non-English files
    list_of_files = os.listdir(transcripts_folder)
    for file in list_of_files:
        if file.startswith('.') and '.json' not in file:
            os.remove(f"{transcripts_folder}/{file}")

    filter_non_english(transcripts_folder)
    correct_user_and_video_ids(users_folder, transcripts_folder)

    # 7. Preprocess the data and save it into a new folder
    preprocess(transcripts_folder)
    correct_user_and_video_ids(users_folder, transcripts_folder)



def download_captions(user_folder, save_to, sleep_time=360):
    """
    Download captions for YouTube videos from a given user folder.

    Args:
        user_folder (str): Path to the folder containing user JSON files.
        save_to (str): Path to the folder where the transcripts will be saved.
        sleep_time (int, optional): Maximum sleep time between requests. Defaults to 360 seconds.

    Returns:
        None
    """
    youtube = YouTubeAPI()
    youtuber_files = list(Path(user_folder).iterdir())

    num_transcripts = 0
    num_videos = 0
    total = len(youtuber_files)

    with tqdm(total=total, desc=f"Getting captions 0/0") as pbar:
        for filepath in youtuber_files:
            # Get the USER ID from the filename
            user_id = filepath.name.split('.')[0]

            # Check if the user already has a file with all the video ids
            with open(filepath, 'r') as f:
                user_data = json.load(f)

                if 'total_videos' in user_data and 'video_ids' in user_data:
                    num_transcripts += len(user_data['video_ids'])
                    num_videos += user_data['total_videos']
                    pbar.update(1)
                    pbar.set_description(f"Getting captions {num_transcripts}/{num_videos}")
                    continue
        
            # Gather all videos from the user across all their channels
            videos = []
            youtube_channels = WikiDataUsers.get_youtube_channel(filepath)
            try:
                for youtube_channel in youtube_channels:
                    videos += youtube.get_channel_videos(youtube_channel)
            except Exception as e:
                print(f"Can't get channel videos from {filepath}")
                continue

            # Download all the transcripts for the videos
            ids = []
            num_videos += len(videos)
            for video in tqdm(videos, desc=f"Processing videos|{user_id}", leave=False, total=len(videos)):
                video_id = video.video_id
                transcript = youtube.get_video_transcript(video)

                if transcript is not None:
                    num_transcripts += 1

                    data = {
                        'video_id': video_id,
                        'video_url': video.watch_url,
                        'video_title': video.title,
                        'video_description': video.description,
                        'is_music': video.is_music,
                        'is_trailer': video.is_trailer,
                        'metadata': video.get_metadata,
                        'transcript': transcript.fetch(),
                    }

                    with open(f"{save_to}/{video_id}.json", 'w') as f:
                        json.dump(data, f, indent=4)

                    ids.append(video_id)

                    sleeptime = random.randint(1, 6)
                    sleep(sleeptime)
                
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            data['total_videos'] = len(videos)
            data['video_ids'] = ids

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)

            pbar.update(1)
            pbar.set_description(f"Getting captions {num_transcripts}/{num_videos}")

            sleeptime = random.randint(30, sleep_time)
            sleep(sleeptime)


def filter_music_and_trailer(transcripts_folder):
    """
    Filter out music videos and movie trailer files from the transcripts folder.

    Args:
        transcripts_folder (str): Path to the folder containing transcript JSON files.

    Returns:
        None
    """
    files = os.listdir(transcripts_folder)
    num_deleted = 0

    with tqdm(files, desc=f"Filtering files (deleted {num_deleted}/{len(files)})") as pbar:
        for file in pbar:
            filepath = os.path.join(transcripts_folder, file)

            with open(filepath, 'r') as f:
                data = json.load(f)

            if data['is_music'] or data['is_trailer']:
                os.remove(filepath)
                num_deleted += 1
                pbar.set_description(f"Filtering files (deleted {num_deleted}/{len(files)})")


def filter_non_english(transcripts_folder):
    """
    Filter out non-English transcript files from the transcripts folder.

    Args:
        transcripts_folder (str): Path to the folder containing transcript JSON files.

    Returns:
        None
    """
    files = os.listdir(transcripts_folder)
    identifier = LangID()
    num_deleted = 0

    with tqdm(files, desc=f"Filtering non-English files (deleted {num_deleted}/{len(files)})") as pbar:
        for file in pbar:
            filepath = os.path.join(transcripts_folder, file)
            lang = identifier.identify_from_transcript(filepath)

            if 'en' not in lang:
                os.remove(filepath)
                num_deleted += 1
                pbar.set_description(f"Filtering non-English files (deleted {num_deleted}/{len(files)})")


def filter_speakers_in_transcripts(user_folder, transcripts_folder, device="cuda:0", token=None):
    """
    Filters out speakers from transcripts in the specified folder.

    Args:
        user_folder (str): The folder containing user data.
        transcripts_folder (str): The folder containing transcript data.
        device (str, optional): The device to use for speaker diarization. Defaults to "cuda:0".
        token (str, optional): The Hugging Face API token. Defaults to None.

    Returns:
        None
    """

    # Initialize YouTube API and Speaker Diarization objects
    youtube_api = YouTubeAPI()
    if "cuda" in device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
        speaker_diarization = SpeakerDiarization(
            device="cuda", hf_auth_token=token,
        )
    else:
        speaker_diarization = SpeakerDiarization(
            device=device, hf_auth_token=token,
        )

    # Get list of transcript files
    user_files = sorted(list(Path(user_folder).iterdir()))
    files = []
    for user_file in user_files:
        with open(user_file, "r") as f:
            user_data = json.load(f)
            if "video_ids" not in user_data:
                continue
            for file in user_data["video_ids"]:
                files.append(Path(f"{transcripts_folder}/{file}.json"))

    # Process each transcript file
    with tqdm(files, desc="Filtering speakers in transcripts") as pbar:
        for file in pbar:
            filename = file.name.split(".")[0]

            # Skip files that don't exist or have already been processed
            if not os.path.isfile(f"{transcripts_folder}/{filename}.json"):
                continue
            if os.path.isfile(f"{transcripts_folder}/.{filename}"):
                continue

            try:
                # Load transcript data
                with open(file, "r") as f:
                    transcript_data = json.load(f)
                video_url = transcript_data["video_url"]
                transcript = transcript_data["transcript"]

                # Get list of additional videos from the same channel
                channel_url = youtube_api.get_channel_from_video(video_url)
                list_of_additional_videos = youtube_api.iterate_channel_videos(
                    channel_url, sort_by=["view_count", "duration"]
                )

                # Filter speakers from the transcript
                pbar.set_postfix({"current video": filename})
                transcript, num_speakers = filter_speaker_out_of_transcript(
                    speaker_diarization,
                    video_url,
                    transcript,
                    list_of_additional_videos,
                    procent=70.0,
                )

                # Update and save transcript data
                transcript_data["transcript"] = transcript
                transcript_data["num_speakers"] = num_speakers
                with open(file, "w") as f:
                    json.dump(transcript_data, f, indent=4)

                # Create a marker file to indicate successful processing
                Path(f"{transcripts_folder}/.{filename}").touch()

            except Exception as e:
                print(f"Error in file: {file}. Skipping for now.")

    # Helper function to filter out a specific speaker from a transcript
    def filter_speaker_out_of_transcript(self, speaker_diarization, video_url, transcript, list_of_additional_videos, procent):
        """
        Filters out the specified speaker from the transcript of a YouTube video.

        Args:
            speaker_diarization (SpeakerDiarization): The speaker diarization object used for analysis.
            video_url (str): The URL of the YouTube video.
            transcript (str): The original transcript of the video.
            list_of_additional_videos (list[str]): A list of additional video URLs from the same channel.
            procent (float, optional): The minimum percentage of the main speaker's presence in the video. Defaults to 70.0.

        Returns:
            tuple[str, int]: The filtered transcript and the number of speakers detected.
        """

        # Download and save the audio from the video
        audio_file = "audio.wav"
        youtube_api.get_audio(video_url, reduce_quality=True, reduced_file=audio_file)

        # Get initial speaker diarization results
        diarization_results = speaker_diarization.get_speaker_diarization(audio_file)
        main_speaker = diarization_results.argmax()
        num_speakers = len(diarization_results.labels())

        # If there is more than one speaker, attempt to identify and filter the main speaker
        if num_speakers != 1:
            concat_audio_file = "concat_audio.wav"
            shutil.copy(audio_file, concat_audio_file)

            for additional_video_url in list_of_additional_videos:
                # Download and concatenate audio from additional videos
                audio_id = str(uuid.uuid4()).split("-")[0]
                extra_audio_file = f"{audio_id}.wav"
                youtube_api.get_audio(
                    additional_video_url, reduce_quality=True, reduced_file=extra_audio_file
                )
                concat_audio_file = SpeakerDiarization.concat_audio(
                    [concat_audio_file, extra_audio_file], concat_audio_file
                )
                os.remove(extra_audio_file)

                # Update speaker diarization results
                diarization_results = speaker_diarization.get_speaker_diarization(
                    concat_audio_file
                )
                main_speaker = diarization_results.argmax()
                speaker_share = speaker_diarization.get_speaker_share(
                    diarization_results
                )

                # Check if the main speaker's share exceeds the specified percentage
                if speaker_share[main_speaker]["duration_procentage"] > procent:
                    main_speaker_timeline = speaker_diarization.get_speaker_annotation(
                        diarization_results, main_speaker
                    )
                    transcript = SpeakerDiarization.extract_speaker_transcript(
                        main_speaker_timeline, transcript
                    )
                    break

            os.remove(concat_audio_file)
        else:
            main_speaker_timeline = speaker_diarization.get_speaker_annotation(
                diarization_results, main_speaker
            )
            transcript = SpeakerDiarization.extract_speaker_transcript(
                main_speaker_timeline, transcript
            )

        os.remove(audio_file)

        return transcript, num_speakers


def preprocess(transcripts_folder):
    """
    Preprocesses transcripts in the specified folder.

    Args:
        transcripts_folder (str): The folder containing transcript data.

    Returns:
        None
    """
    for file in Path(transcripts_folder).iterdir():
        if ".json" not in file.name:
            continue

        with open(file, "r") as f:
            data = json.load(f)

        # Apply preprocessing to the transcript
        transcript = Preprocessing.filter(data["transcript"])

        # Update the transcript data
        data["transcript"] = transcript
        with open(file, "w") as f:
            json.dump(data, f, indent=4)




def filter_empty_transcripts(transcripts_folder):
    """
    Filter out empty transcript files from the given transcripts folder.

    Args:
        transcripts_folder (str): Path to the folder containing transcript JSON files.

    Returns:
        None
    """
    empty_files = []

    for file in Path(transcripts_folder).iterdir():
        if not file.name.endswith('.json'):
            continue

        with open(file, 'r') as f:
            data = json.load(f)

        if not data['transcript']:
            empty_files.append(file)

    for file in empty_files:
        os.remove(file)

def correct_user_and_video_ids(users_folder, transcript_folder):
    """
    Corrects user and video IDs in the user files after filtering.

    Args:
        users_folder (str): The folder containing user data.
        transcript_folder (str): The folder containing transcript data.

    Returns:
        None
    """

    # Helper functions for user and transcript management
    def get_list_of_users():
        """
        Gets the list of users and their filenames.

        Returns:
            tuple[list, list]: A tuple containing the list of users and their filenames.
        """
        users = []
        filenames = os.listdir(users_folder)
        for filename in filenames:
            if filename.endswith(".json"):
                with open(os.path.join(users_folder, filename)) as f:
                    users.append(json.load(f))
        return users, filenames

    def removal_script(check, users, filenames, folder_path):
        """
        Removes files and updates user data based on a check list.

        Args:
            check (list): A list of boolean values indicating which items to remove.
            users (list): The list of users.
            filenames (list): The list of filenames.
            folder_path (str): The path to the folder containing the files.

        Returns:
            tuple[list, list]: A tuple containing the updated list of users and filenames.
        """
        indexes = [i for i, x in enumerate(check) if not x]
        for i in sorted(indexes, reverse=True):
            os.remove(os.path.join(folder_path, filenames[i]))
            users.pop(i)
            filenames.pop(i)
        return users, filenames

    # Remove users that have no transcripts
    def has_transcripts(user):
        """
        Checks if a user has transcripts.

        Args:
            user (dict): The user data.

        Returns:
            bool: True if the user has transcripts, False otherwise.
        """
        return "video_ids" in user and len(user["video_ids"]) > 0

    # Main function logic
    users, filenames = get_list_of_users()

    # Remove users with no transcripts
    check = [has_transcripts(user) for user in users]
    if not all(check):
        print(f"There are {len(check) - sum(check)} users with no transcripts in the dataset")
        print("Removing users...")
        users, filenames = removal_script(check, users, filenames, users_folder)

    # Get list of transcripts and their filenames
    def get_transcripts():
        """
        Gets the list of transcripts and their filenames.

        Returns:
            tuple[list, list]: A tuple containing the list of transcripts and their filenames.
        """
        transcripts = []
        filenames = os.listdir(transcript_folder)
        for filename in filenames:
            if filename.endswith(".json"):
                with open(os.path.join(transcript_folder, filename)) as f:
                    transcripts.append(json.load(f))
        return transcripts, filenames

    transcripts, t_filenames = get_transcripts()

    # Remove users with missing transcripts
    def is_transcript_exist(user, t_filenames):
        """
        Checks if all transcripts for a user exist.

        Args:
            user (dict): The user data.
            t_filenames (list): The list of transcript filenames.

        Returns:
            tuple[bool, list]: A tuple containing a boolean indicating if all transcripts exist, and a list of existence flags.
        """
        transcripts = user["video_ids"]
        exists = [transcript + ".json" in t_filenames for transcript in transcripts]
        return all(exists), exists

    checks = [is_transcript_exist(user, t_filenames) for user in users]
    if not all([check[0] for check in checks]):
        print(f"There are {len(checks) - sum([check[0] for check in checks])} users with missing transcripts")
        print("Updating or removing users...")

        all_checks = []
        for i, user, check in zip(range(len(users)), users, checks):
            _check, exists = check

            # If there are missing transcripts, but not all are missing
            if not _check and all_checks[-1]:
                # Update the user's video_ids list
                user["video_ids"] = [
                    transcript
                    for transcript, exist in zip(user["video_ids"], exists)
                    if exist
                ]

                # Update the user JSON file
                with open(os.path.join(users_folder, filenames[i]), "w") as f:
                    json.dump(user, f, indent=4)

        # Remove users with no transcripts
        users, filenames = removal_script(all_checks, users, filenames, users_folder)

    # Get list of transcript names from users
    def get_transcript_names_from_users(users):
        """
        Gets the list of transcript names from the users.

        Args:
            users (list): The list of users.

        Returns:
            list: A list of transcript names.
        """
        transcripts = []
        for user in users:
            transcripts += user["video_ids"]
        transcripts = list(set(transcripts))
        return transcripts

    actual_transcripts = get_transcript_names_from_users(users)

    # Remove transcripts that don't belong to any user
    check = [transcript["video_id"] in actual_transcripts for transcript in transcripts]
    if not all(check):
        print(f"There are {len(check) - sum(check)} transcripts that do not belong to any user")
        print("Removing transcripts...")
        transcripts, t_filenames = removal_script(check, transcripts, t_filenames, transcript_folder)

    # Remove empty transcripts
    def is_not_transcript_empty(transcript):
        """
        Checks if a transcript is empty.

        Args:
            transcript (dict): The transcript data.

        Returns:
            bool: True if the transcript is not empty, False otherwise.
        """
        return len(transcript["transcript"]) != 0

    check = [is_not_transcript_empty(transcript) for transcript in transcripts]
    if not all(check):
        print(f"There are {len(check) - sum(check)} empty transcripts in the dataset")
        print("Removing empty transcripts...")
        transcripts, t_filenames = removal_script(check, transcripts, t_filenames, transcript_folder)


if __name__ == '__main__':
    main()