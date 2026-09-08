import streamlit as st
import librosa
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Guitar Chords Extractor", page_icon="🎸", layout="centered")

st.title("🎸 Guitar Chords Extractor")
st.markdown("Upload your song and instantly get the chords to play along! 🎶")

def get_chord_templates():
    templates = []
    chord_names = []
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i in range(12):
        # Major
        template_maj = np.zeros(12)
        template_maj[i] = 1
        template_maj[(i + 4) % 12] = 1
        template_maj[(i + 7) % 12] = 1
        templates.append(template_maj)
        chord_names.append(notes[i])
        
        # Minor
        template_min = np.zeros(12)
        template_min[i] = 1
        template_min[(i + 3) % 12] = 1
        template_min[(i + 7) % 12] = 1
        templates.append(template_min)
        chord_names.append(notes[i] + 'm')
        
    return np.array(templates), chord_names

uploaded_file = st.file_uploader("Upload an Audio File (MP3, WAV)", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    with st.spinner("Analyzing chords... Please wait a moment ⏳"):
        # Save to temp file because librosa needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            y, sr = librosa.load(tmp_path)
            
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0]
            
            st.success(f"🎵 Estimated Tempo: {tempo:.0f} BPM")
            
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
            
            templates, chord_names = get_chord_templates()
            
            beat_chords = []
            for i in range(chroma_sync.shape[1]):
                chroma_beat = chroma_sync[:, i]
                correlations = np.dot(templates, chroma_beat)
                beat_chords.append(chord_names[np.argmax(correlations)])
                
            beats_per_bar = 4
            bar_chords = []
            for i in range(0, len(beat_chords), beats_per_bar):
                bar_beats = beat_chords[i:i+beats_per_bar]
                most_frequent = max(set(bar_beats), key=bar_beats.count)
                bar_chords.append(most_frequent)
                
            st.subheader("🎼 Detected Chords (4 Bars per line)")
            
            bars_per_line = 4
            last_chord = None
            
            chords_output = ""
            for i in range(0, len(bar_chords), bars_per_line):
                line_bars = bar_chords[i:i+bars_per_line]
                line_str = ""
                for chord in line_bars:
                    if chord == last_chord:
                        line_str += "|  -  "
                    else:
                        line_str += f"| {chord.ljust(3)} "
                    last_chord = chord
                line_str += "|\n"
                chords_output += line_str
                
            st.code(chords_output, language="text")
            
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            os.remove(tmp_path)
