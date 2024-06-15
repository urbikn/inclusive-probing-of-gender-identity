from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Segment
import torchaudio
import numpy as np
import torch
import wave
import onnxruntime as ort

# Set the logging level to ERROR to ignore warnings
ort.set_default_logger_severity(3)

class SpeakerDiarization():
    def __init__(self, device='cuda', hf_auth_token=None):
        # Initialize speaker diarization pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_auth_token
        )
        self.pipeline.to(torch.device(device))
        self.pipeline.embedding_batch_size = 64 + 32

        # Set segment precision
        Segment.set_precision(2)
    
    def get_speaker_diarization(self, filepath):
        """ Get speaker diarization result from the given audio file using pipeline.

        Args:
            filepath (str): File path of the audio file
        
        Returns:
            speaker annotation (pyannote.core.annotation.Annotation): Speaker diarization result
        """
        waveform, sample_rate = torchaudio.load(filepath)
        with ProgressHook(transient=True) as hook: # Transient=True to hide the progress bar after completion
            results = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=1, max_speakers=5, hook=hook)
        
        return results

    def get_speaker_share(self, diarization_results):
        """ Get speaker share from the given speaker diarization result.

        Args:
            diarization_results (pyannote.core.annotation.Annotation): Speaker diarization result

        Returns:
            dict (speaker_name: {duration, duration_procentage}): Speaker share
        """
        total_duration = diarization_results.get_timeline().duration()

        speaker_share = {}
        for speaker_label in diarization_results.labels():
            duration = diarization_results.label_duration(speaker_label)
            duration_procentage = int(duration/total_duration*100)
            speaker_share[speaker_label] = {'duration': duration, 'duration_procentage': duration_procentage}
        
        return speaker_share

    def get_speaker_annotation(self, diarization_results, speaker_name, remove_overlays=True):
        """ Get speaker annotation from the given speaker diarization result.

        Args:
            diarization_results (pyannote.core.annotation.Annotation): Speaker diarization result
            speaker_name (str): Speaker name
            remove_overlays (bool, optional): Remove overlays of main speaker's annotation timeline. Defaults to True.
        
        Returns:
            pyannote.core.annotation.Annotation: Speaker annotation
        """
        # Get the main speaker's annotation timeline
        main_speaker_timeline = diarization_results.label_timeline(speaker_name).support(0.8) # support(0.8) to join segments that are apart by 0.8 seconds or less

        # Remove overlays of main speaker's annotation timeline
        if remove_overlays:
            for speaker in diarization_results.labels():
                if speaker == speaker_name:
                    continue

                speaker_timeline = diarization_results.label_timeline(speaker).support(0.5) 
                # Remove overlays
                main_speaker_timeline = main_speaker_timeline.extrude(speaker_timeline)

        return main_speaker_timeline.support(0.8)

    @staticmethod
    def concat_audio(audio_files, output_file):
        """ Concatenate audio `wav` files.

        Args:
            audio_files (list): List of audio files to concatenate. Defaults to [].
            output_file (str): Output file path to save the concatenated audio.
        
        Returns:
            str: File path of the output_file
        """
        combined = AudioSegment.empty()
        for file in audio_files:
            combined += AudioSegment.from_file(file, format="wav")
        combined.export(output_file, format="wav")

        return output_file

    @staticmethod
    def extract_speaker_transcript(speaker_annotation, transcript):
        """ Extracts the transcript corresponding to a particular speaker.

        Args:
            speaker_annotation (pyannote.core.annotation.Annotation): 
                The speaker annotations which include segments where the speaker is speaking.
            transcript (list): 
                A list of transcript chunks, where each chunk is a dictionary containing 
                'start', 'duration', and 'text' of each spoken segment in the entire video.

        Returns:
            transcript_filtered (list): 
                A list of transcript chunks that only includes segments where the input speaker is speaking.
        """

        transcript_filtered = []

        # Iterate over each transcript chunk
        for transcript_chunk in transcript:
            # Create a Segment object for the current transcript chunk
            segment = Segment(np.round(transcript_chunk['start'], 2), np.round(transcript_chunk['start'] + transcript_chunk['duration'], 2))


            # For each chunk, iterate over all segments of the speaker
            for speaker_segment in speaker_annotation:

                speaker_segment = Segment(np.round(speaker_segment.start, 2), np.round(speaker_segment.end, 2))
                # Checks if the current speaker segment intersects with the transcript chunk segment
                # If it intersects, it means the speaker was speaking during this transcript chunk
                if segment in speaker_segment:
                    # Add the intersecting transcript chunk to the filtered transcript
                    transcript_filtered.append(transcript_chunk)
                    # Break the loop as we've found an intersecting segment, no need to check the rest
                    break

        return transcript_filtered