import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import yt_dlp
import scipy.signal
from sklearn.cluster import KMeans

st.set_page_config(page_title="Guitar Chords Extractor", page_icon="🎸", layout="centered")

st.title("🎸 Guitar Chords Extractor (Pro)")
st.markdown("Upload your song OR paste a YouTube link to instantly get the chords, organized by song sections! 🎶")

def get_chord_templates():
    templates = []
    chord_names = []
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i in range(12):
        template_maj = np.zeros(12)
        template_maj[i] = 1
        template_maj[(i + 4) % 12] = 1
        template_maj[(i + 7) % 12] = 1
        templates.append(template_maj)
        chord_names.append(notes[i])
        
        template_min = np.zeros(12)
        template_min[i] = 1
        template_min[(i + 3) % 12] = 1
        template_min[(i + 7) % 12] = 1
        templates.append(template_min)
        chord_names.append(notes[i] + 'm')
    return np.array(templates), chord_names

def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

def process_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        
        st.write("Detecting tempo and beats...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
            
        st.success(f"🎵 Estimated Tempo: {tempo:.0f} BPM")
        
        st.write("Analyzing song structure (Intro, Chorus, Verse)...")
        # Extract MFCCs for structure (timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames)
        
        # Cluster beats into 4 structural sections
        n_sections = 4
        kmeans = KMeans(n_clusters=n_sections, random_state=42, n_init=10)
        beat_labels = kmeans.fit_predict(mfcc_sync.T)
        
        # Smooth the labels to avoid rapid changes
        beat_labels_smoothed = scipy.signal.medfilt(beat_labels, kernel_size=15).astype(int)
        
        seen_clusters = []
        def get_section_name(cluster_id):
            if cluster_id not in seen_clusters:
                seen_clusters.append(cluster_id)
            idx = seen_clusters.index(cluster_id)
            names = ["Part A (Intro/Verse)", "Part B (Chorus)", "Part C (Bridge/Solo)", "Part D (Outro)"]
            if idx < len(names):
                return names[idx]
            return f"Part {idx+1}"
            
        st.write("Extracting chords...")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        
        templates, chord_names = get_chord_templates()
        beat_chords = []
        for i in range(chroma_sync.shape[1]):
            chroma_beat = chroma_sync[:, i]
            correlations = np.dot(templates, chroma_beat)
            beat_chords.append(chord_names[np.argmax(correlations)])
            
        st.subheader("🎼 Detected Chords & Song Structure")
        
        beats_per_bar = 4
        bars_per_line = 4
        
        chords_output = ""
        current_section = -1
        line_str = ""
        bars_in_current_line = 0
        last_chord = None
        
        for i in range(0, len(beat_chords), beats_per_bar):
            bar_beats = beat_chords[i:i+beats_per_bar]
            if not bar_beats:
                continue
                
            chord = max(set(bar_beats), key=bar_beats.count)
            
            bar_labels = beat_labels_smoothed[i:min(i+beats_per_bar, len(beat_labels_smoothed))]
            bar_section = max(set(bar_labels), key=list(bar_labels).count) if len(bar_labels) > 0 else current_section
            
            if bar_section != current_section:
                if line_str:
                    chords_output += line_str + "|\n\n"
                    line_str = ""
                    bars_in_current_line = 0
                    
                current_section = bar_section
                section_name = get_section_name(current_section)
                chords_output += f"--- {section_name} ---\n"
            
            if chord == last_chord:
                line_str += "|  -  "
            else:
                line_str += f"| {chord.ljust(3)} "
                
            last_chord = chord
            bars_in_current_line += 1
            
            if bars_in_current_line == bars_per_line:
                chords_output += line_str + "|\n"
                line_str = ""
                bars_in_current_line = 0
                
        if line_str:
            chords_output += line_str + "|\n"
            
        st.code(chords_output, language="text")
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")

tab1, tab2 = st.tabs(["📁 Upload File", "🔗 YouTube Link"])

with tab1:
    uploaded_file = st.file_uploader("Upload an Audio File (MP3, WAV)", type=["mp3", "wav", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        with st.spinner("Analyzing chords and structure... Please wait a moment ⏳"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            process_audio(tmp_path)
            os.remove(tmp_path)

with tab2:
    yt_url = st.text_input("Paste YouTube URL here:")
    if st.button("Extract Chords from YouTube"):
        if yt_url:
            with st.spinner("Downloading audio from YouTube... This may take a minute ⏳"):
                try:
                    yt_audio_path = download_youtube_audio(yt_url)
                    st.success("Download complete! Now analyzing...")
                    with st.spinner("Analyzing chords and structure... Please wait a moment ⏳"):
                        process_audio(yt_audio_path)
                    if os.path.exists(yt_audio_path):
                        os.remove(yt_audio_path)
                except Exception as e:
                    st.error(f"Failed to process YouTube link: {e}")
        else:
            st.warning("Please enter a valid YouTube URL.")
