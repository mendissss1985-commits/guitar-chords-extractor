# Guitar Chords Extractor

This is a simple Python project to extract guitar chords (Major and Minor) from an audio file using Chroma features.

## Requirements

You need `librosa` and `numpy` installed. You can install them via pip:

```bash
pip install -r requirements.txt
```

## Usage

Place any `.mp3` or `.wav` song in this folder, and run the script with the path to the song:

```bash
python chord_extractor.py "your_song.mp3"
```

The script will analyze the frequencies and output the sequence of chords detected!
