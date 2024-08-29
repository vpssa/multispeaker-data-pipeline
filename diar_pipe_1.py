import os
from pytube import YouTube
from pydub import AudioSegment
import openpyxl
import moviepy.editor as mp
import pandas as pd

def download_and_combine_audio(row, output_root_folder):
    link_columns = ["Video Link 2", "Video Link 3", "Video Link 4", "Video Link 5"]

    # Extract video links from the row
    links = [row[column] for column in link_columns]
    links = [link for link in links if not pd.isna(link)]

    video_links = [row['Video Link 1']] + links

    # Initialize combined audio for this row
    combined_audio = AudioSegment.silent(duration=0)

    for link_index, link in enumerate(video_links):
        yt = YouTube(link)
        youtube_object = yt.streams.get_lowest_resolution()
        youtube_object.download(filename=f"speaker{link_index}.mp4")
        clip = mp.VideoFileClip(f"speaker{link_index}.mp4")
        clip.audio.write_audiofile(f"speaker{link_index}_.wav")
        os.remove(f"speaker{link_index}.mp4")

        # Trim the first and last 10 seconds of audio
        trim_duration = 10000
        sound = AudioSegment.from_wav(f"speaker{link_index}_.wav")
        sound = sound[trim_duration:-trim_duration]

        # Add 2 seconds of silence between each audio clip (except the first one)
        if link_index > 0:
            silence = AudioSegment.silent(duration=2000)
            combined_audio += silence

        combined_audio += sound

    # Create the root folder if it doesn't exist
    os.makedirs(output_root_folder, exist_ok=True)

    # Export the combined audio for this row
    output_filename = os.path.join(output_root_folder, f"speaker_{a}.wav")
    combined_audio.export(output_filename, format="wav")

    return output_filename

if __name__ == "__main__":
    excel_path = "test1.xlsx"  # Replace with the path to your Excel file
    output_root_folder = "diarization"  # Replace with the desired output folder

    # Read Excel file
    data = pd.read_excel(excel_path)
    a=0
    # Iterate over rows in the Excel file
    for _, row in data.iterrows():
        # Apply the function to each row of the DataFrame
        result = download_and_combine_audio(row, output_root_folder)
        print("Result : ",result)
        a=a+1

    print("Script execution completed.")

