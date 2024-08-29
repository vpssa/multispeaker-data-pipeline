from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import os
from pathlib import Path

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipeline.to(device)

input_parent_folder = "/home/azureuser/output_folder"  # Provide the path to the parent folder containing folders with WAV files

# Iterate over each folder in the parent folder
for foldername in os.listdir(input_parent_folder):
    folder_path = os.path.join(input_parent_folder, foldername)
    
    if os.path.isdir(folder_path):
        # Iterate over each WAV file in the current folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(folder_path, filename)

                # Load the audio file
                audio = AudioSegment.from_file(audio_path)

                # Apply pretrained pipeline
                diarization = pipeline(audio_path)

                # Initialize an empty audio segment for the final output
                final_audio = AudioSegment.empty()

                # Initialize a dictionary to store the total duration for each speaker
                speaker_durations = {}

                # Extract segments for each speaker and calculate their total duration
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    duration = turn.end - turn.start
                    if speaker in speaker_durations:
                        speaker_durations[speaker] += duration
                    else:
                        speaker_durations[speaker] = duration

                # Find the dominant speaker
                dominant_speaker = max(speaker_durations, key=speaker_durations.get)

                # Create a folder for the output if it doesn't exist
                output_folder = os.path.join("output_folders", foldername)  # Remove '.wav' extension from filename
                os.makedirs(output_folder, exist_ok=True)

                # Extract segments for the dominant speaker and save them in the corresponding output folder
                start = 0
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker != dominant_speaker:
                        stop = turn.start * 1000  # convert to milliseconds
                        segment = audio[start:stop]
                        final_audio += segment
                        start = turn.end * 1000  # convert to milliseconds

                # Add the last segment
                final_audio += audio[start:]
                
                # Export the final audio to the output folder
                output_path = os.path.join(output_folder, filename)
                final_audio.export(output_path, format="wav")

                print(f"Processing completed for {filename}. Result saved in {output_folder}")
