import os
import subprocess
import io
from pathlib import Path
import select
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
from pyannote.audio import Pipeline
import torch
import shutil

from pydub import AudioSegment
from pydub.silence import split_on_silence

def setup_project(input_path):
    if not input_path.endswith(".wav"):
        raise ValueError("Invalid file format. .wav required.")
    video_id = os.path.basename(input_path).split(".")[0]
    # print("video_id: ", video_id)
    try:
        Path(video_id, "output").mkdir(parents=True, exist_ok=True)
        Path(video_id, "input").mkdir(parents=True, exist_ok=True)
        out_dir = Path(video_id, "input", "file.wav")
        shutil.copy(input_path, out_dir)
    except OSError as e:
        print(f"Failed to setup project: {e}")
        return None
    return video_id

def find_files(in_path, extensions):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    # print("file: ", file)
    return out

def separate(inp=None, outp=None, model=None, extensions=None):
    inp = inp or in_path
    outp = outp or out_path
    cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", model]
    files = [str(f) for f in find_files(inp, extensions)]
    if not files:
        print(f"No valid audio files in {inp}")
        return
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    print("Separation command output:", stdout.decode())
    print("Separation command error:", stderr.decode())
    if p.returncode != 0:
        print("Separation command failed, something went wrong.")

def diarize_and_save(input_path, hf_token):
    # print("=============================================================================")

    project_dir = setup_project(input_path)

    # project_dir = os.path.basename(input_path).split(".")[0]
    # print("project_dir: ",project_dir)

    output_dir = os.path.join("output_folder", project_dir)
    # print("output_dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(project_dir, "output", "htdemucs", "file", "vocals.wav")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

    pipeline.to(torch.device("cuda"))
    diarization = pipeline(file_path)
    audio = AudioSegment.from_wav(file_path)

    speakers = []
    buffer = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = int(turn.start * 1000)
        end = int(turn.end * 1000)

        overlap = False
        for s, segments in buffer.items():
            if any(segment[1] > start and segment[0] < end for segment in segments):
                overlap = True
                break

        if not overlap:
            speakers.append(speaker)
            buffer.setdefault(speaker, []).append((start, end))
  
    silence = AudioSegment.silent(duration=2000)  # 2 seconds of silence
    for s, segments in buffer.items():
        buffer_audio = sum((audio[start:end] + silence) for start, end in segments)
        # buffer_audio.export(os.path.join(".", "output_folder", s + "_" + input_path.split(".")[0] + ".wav"), format="wav")
        output_filename = f"{s}_{os.path.basename(input_path).split('.')[0]}.wav"
        # print("output_filename: ", output_filename)
        # print("output_dir: ", output_dir)
        buffer_audio.export(os.path.join(output_dir, output_filename), format="wav")
        
        # buffer_audio.export(os.path.join( ".","output_folder"), format="wav")
        # buffer_audio.export(os.path.join(".", "output_folder", output_filename), format="wav")
    return speakers
    
def cleanup(project_id):
    input_dir = os.path.join(".", project_id, "input")
    output_dir = os.path.join(".", project_id, "output")

    # Remove files in input directory
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Remove files in output directory
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Remove input and output directories
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)

    # Remove project directory
    # os.rmdir(os.path.join(".", project_id))



def cleanup(project_id):
    shutil.rmtree(os.path.join(".", project_id, "input"))
    shutil.rmtree(os.path.join(".", project_id, "output"))
    os.rmdir(os.path.join(".", project_id))




def process_directory(directory, hf_token):
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            input_path = os.path.join(directory, file)
            # print("Processing file:", input_path)
            project_id = setup_project(input_path)
            # print("project_id: ", project_id)
            if project_id is None:
                continue

            model = "htdemucs"
            extensions = ["wav"]

            in_path = os.path.join(".", project_id, "input")
            out_path = os.path.join(".", project_id, "output")

            # print("Separating files in:", in_path)
            # print("Separating files:", find_files(in_path, extensions))
            separate(inp=in_path, outp=out_path, model=model, extensions=extensions)
            

            # print("Diarizing file:", file)
            speakers = diarize_and_save(input_path, hf_token)
            cleanup(project_id)

if __name__ == "__main__":
    directory = 'diarization'  # replace with your directory
    # print("directory: ", directory)
    hf_token  = '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'

    process_directory(directory, hf_token)
