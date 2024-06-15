from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import NoTranscriptAvailable, NoTranscriptFound
from pydub import AudioSegment
from tqdm import tqdm
from pytube import extract, Channel, YouTube
import time
import random
import yt_dlp
import os
import json
import operator
import logging
import pandas as pd
from time import sleep
from functools import cached_property

logger = logging.getLogger(__name__)

class YouTubeVideo(YouTube):
    def __init__(self, description = None, ydl=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._description = description

        # For general ifnormation about the video
        if ydl is None:
            self.ydl = yt_dlp.YoutubeDL({
                'cachedir': "__pycache__",
                'download_archive': "__pycache__/downloaded_videos.txt",
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'process': False,
            })
        else:
            self.ydl = ydl

    # Helper function to extract video IDs from a YouTube channel
    def __find_values(self, id, json_repr):
        results = []

        def _decode_dict(a_dict):
            try: results.append(a_dict[id])
            except KeyError: pass
            return a_dict

        json.loads(json_repr, object_hook=_decode_dict)  # Return value ignored.
        return results

    @cached_property
    def get_title(self) -> str:
        """Get title of video.

        Returns:
            str: Title of video.
        """
        # Check if self.title exists as a property
        if self.title is not None:
            return self.title
        else:
            return ""

    @cached_property
    def get_metadata(self, default=None) -> dict:
        """Get metadata of video. 
        This includes views, likes, comments, subscribers, upload date, duration, duration string and tags.

        Returns:
            dict: Dictionary with metadata.
        """
        try:
            video_data = self.ydl.extract_info(self.watch_url)
        except Exception as e:
            print(f"Failed to get metadata: {e}")
            return default

        metadata = {
            'views': video_data.get('view_count', 0),
            'likes': video_data.get('like_count', 0),
            'comments': video_data.get('comment_count', 0),
            'subscribers': video_data.get('channel_follower_count', 0),
            'upload_date': video_data.get('upload_date', 0),
            'duration': video_data.get('duration', 0),
            'duration_string': video_data.get('duration_string', ""),
            'tags': video_data.get('tags', []),
        }

        return metadata


    @cached_property
    def description(self) -> str:
        """Get the video description.

        Returns:
            str: The video description.
        """
        if self._description is not None:
            return self._description
        else:
            json_dump = json.dumps(self.initial_data)
            description = self.__find_values('attributedDescription', json_dump)

            if len(description) == 0:
                return ""
            
            return description[0].get('content')

    @cached_property
    def is_music(self) -> bool:
        """Check if the video is music.

        Returns:
            bool: True if the video is music, otherwise False.
        """
        json_dump = json.dumps(self.vid_info)
        is_music_video = len(self.__find_values('musicVideoType', json_dump)) != 0

        title = self.get_title.lower()

        # check if music video in title (if not in video info)
        is_music_video = is_music_video or 'music video' in title

        # Check if the video is a cover by checking if the title contains the word 'cover'
        # (not the best indicator, but it's the best we can do for now)
        is_cover = 'cover' in title

        return is_music_video or is_cover
    
    @cached_property
    def is_trailer(self) -> bool:
        """ Check if the video is a trailer.

        Returns:
            bool: True if the video is a trailer, otherwise False.
        """

        # A very simple check to see if the video is a trailer
        if 'trailer' in self.get_title.lower():
            return True
        else:
            return False


class YouTubeAPI:
    def __init__(self):
        # Initialize youtube-dl for downloading transcripts
        self.ydl_transcript = yt_dlp.YoutubeDL({
                'cachedir': "__pycache__",
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            })

        # Initialize youtube-dl for downloading audio
        self.tmp_file = 'tmp_audio_download.wav'
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '5',  # choose audio quality (between 0 (better) and 9 (worse))
            }],
            'outtmpl': self.tmp_file.split('.')[0],  # output filename
            'quiet': True,
            'no_warnings': True,
        }
        self.ydl_audio = yt_dlp.YoutubeDL(ydl_opts)

        # For general ifnormation about the video
        self.ydl = yt_dlp.YoutubeDL({
                'cachedir': "__pycache__",
                'download_archive': "__pycache__/downloaded_videos.txt",
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'process': False,
        })

        self.cache = {}


    def get_video_transcript(self, video):
        """
        Retrieves the transcript for a given YouTube video.

        Args:
            YouTubeVideo: The YouTube video object.

        Returns:
            pytube.Transcript: A transcript object if the transcript is found, otherwise None.
        """
        logger.info(f"Retrieving transcript for video {video.watch_url}")

        # Get the transcript for the video
        transcript = None
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video.video_id)
            if transcript_list is None:
                logger.exception(f"No transcript found for video {video.watch_url}")
                raise NoTranscriptFound(video.video_id)

            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
            if transcript is None:
                logger.exception(f"No manually created transcript found for video {video.watch_url}")
                raise NoTranscriptAvailable(video.video_id)

        except:
            pass

        return transcript

    def get_channel_from_video(self, video_url):
        """
        Retrieves the channel URL from a given YouTube video.

        Args:
            video_url (str): The URL of the YouTube video.
        
        Returns:
            str: The URL of the YouTube channel.
        """

        video_data = self.ydl.extract_info(video_url)
        channel_url = video_data['channel_url'] 
        return channel_url


    def iterate_channel_videos(self, channel_url, sort_by=[], filter_out=True):
        """
        Iterates over all videos uploaded to a given YouTube channel.

        Args:
            channel_url (str): The URL of the YouTube channel.
            sort_by (list[str], optional): 
                List of fields to sort by. 
                Possible values are: `['duration', 'view_count']`.
                Defaults to [] (no sorting).

                'duration' -> will round to the nearest 15 minutes
                'view_count' -> will round to the nearest 1000 views
                    -> the reason for this is to bucket similar videos together
                    -> and to make the sorting more stable
            filter_out (bool, optional):
                If True, will filter out videos that are trailers or music videos.
                Defaults to True.
        
        Returns:
            list[str]: A list of video urls.
        """
        # To be able to extract information about the videos in the channel
        # we need to add `/videos` to the URL
        if 'videos' not in channel_url:
            channel_url += '/videos'
        
        # Check if the channel is already in the cache
        if channel_url in self.cache:
            return self.cache[channel_url]

        # Get information about the channel, including all videos
        data = self.ydl.extract_info(channel_url, process=False)

        entries = list(data['entries'])

        # Get all the video urls, their duration and view count
        videos = {}
        for i, entry in tqdm(enumerate(entries), desc="Processing list of all videos", total=len(entries), leave=False):
            url = entry['url']

            video = YouTubeVideo(url=entry['url'], description=entry['description'], ydl=self.ydl)
            if filter_out and (video.is_music or video.is_trailer):
                continue

            # Round the duration to the nearest 15 minute
            duration = entry.get('duration', 0)
            if duration is None:
                duration = 0

            duration = round(duration / 900) * 900

            # Round the view count to the nearest 1000
            view_count = entry.get('view_count', 0)
            if view_count is None:
                view_count = 0

            view_count = round(view_count / 1000) * 1000

            videos[url] = {
                'url': url,
                'duration': duration,
                'view_count': view_count,
            }

            if i % 100 == 0:
                time_sleep = round(random.uniform(0.5, 1.5), 2)
                time.sleep(time_sleep)


        
        # Create DataFrame
        videos_df = pd.DataFrame(videos).transpose()

        # Sort by duration and view count
        videos_df = videos_df.sort_values(sort_by, ascending=False)

        list_of_urls = videos_df['url'].to_list()

        # Add to cache
        self.cache[channel_url] = list_of_urls

        # Create a generator to iterate over the video urls
        return list_of_urls


    def get_channel_videos(self, channel_url):
        """
        Retrieves the video IDs of all videos uploaded to a given YouTube channel.

        Args:
            channel_url (str): The URL of the YouTube channel.

        Returns:
            list[YouTubeVideo]: a list of YouTube video objects.
        """

        # First check if sufix `/videos` is present in the URL
        if not channel_url.endswith('/videos'):
            channel_url += '/videos'

        # Sometimes, for example when the channel doesn't have any videos tab, it will throw an exception
        try:
            response = self.ydl_transcript.extract_info(channel_url, process=False)
        except:
            return []

        videos = []
        for entry in response['entries']:
            videos.append(YouTubeVideo(url=entry['url'], description=entry['description']))

        return videos


    def __reduce_quality(self, input_file, output_file, frame_rate=22050, sample_width=2, channels=1):
        """ Reduce quality to make the audio file size smaller.

        Args:
            input_file (str): Input file path of the audio (only wav format)
            output_file (str): Output file path to save the processed audio (only wav format)
            frame_rate (int, optional): Frame rate. Defaults to 22050.
            sample_width (int, optional): Sample width. Defaults to 2.
            channels (int, optional): Channels. Defaults to 1 (mono).
        
        Returns:
            None
        """
        audio = AudioSegment.from_file(input_file, format="wav")

        # Reduce quality to make the file size smaller
        audio = audio.set_frame_rate(frame_rate)  # decrease the frame rate
        audio = audio.set_sample_width(sample_width)  # decrease the sample width

        # Convert to mono
        audio = audio.set_channels(channels)

        # Save the processed audio
        audio.export(output_file, format="wav")
    
    def get_audio(self, url, reduce_quality=True, reduced_file=None):
        """ Download audio from the given url.

        Args:
            url (str): Youtube video url
            reduce_quality (bool, optional): Reduce quality to make the audio file size smaller. Defaults to True.
            reduced_file (str, optional): Output file path to save the processed audio (only wav format). Defaults to None (will overwrite the downloaded file).
        
        Returns:
            str: File path of the downloaded audio file
        """

        # Downloads the audio file from the url and saves as `self.tmp_file`
        self.ydl_audio.download([url])
        filepath = self.tmp_file

        if reduce_quality:
            # In case no output file is specified, overwrite the input file
            if reduced_file is None:
                reduced_file = self.tmp_file

            # Reduce quality to make the file size smaller
            self.__reduce_quality(self.tmp_file, reduced_file)
            filepath = reduced_file
        
        return filepath

if __name__ == '__main__':
    channel_id = "https://www.youtube.com/@KatyPerry"

    # Start getting captions
    print('Extracting captions from YouTube channel')
    youtube = YouTubeAPI()

    videos = youtube.get_channel_videos(channel_id)

    # Create a directory to save the transcripts
    if not os.path.exists('transcripts'):
        os.makedirs('transcripts')

    video_ids = []
    for video in tqdm(videos, leave=False, total=len(videos)):
        video_id = video.video_id
        transcript = youtube.get_video_transcript(video)

        if transcript is not None:
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

            if video.is_music or video.is_trailer:
                continue

            # Save into a JSON file
            with open(f"transcripts/{video_id}.json", 'w') as f:
                json.dump(data, f, indent=4)

            # Save video id
            video_ids.append(video_id)

            sleeptime = random.randint(1, 6)
            sleep(sleeptime)
            