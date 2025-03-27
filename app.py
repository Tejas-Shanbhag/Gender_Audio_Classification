import streamlit as st

import pyaudio
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

st.set_page_config(layout='wide')

st.markdown("""
    <style>
    .stForm {
        background-color: #ADD8E6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Audio Classification App</h1>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")

@st.cache_resource
def load_model():
    model_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    model.eval()

    return model,feature_extractor

model,feature_extractor = load_model()


# Audio settings
sampling_rate = 16000  # Sample rate in Hz
channels = 1           # Mono audio
chunk = 1024           # Buffer size
record_seconds = 1     # Duration of recording

# Initialize PyAudio
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, channels=channels,
                      rate=sampling_rate, input=True,
                      frames_per_buffer=chunk)


fig1,fig2 = st.columns(2)


with fig1:
   with st.container(height=600):
    st.image('assets/image.png',use_container_width=True)


with fig2:
    with st.container(height=600,border=True):
        st.write("Click to start the Audio")
        listen = st.button("Start Listening",type='primary')

        st.write("Click to End the Audio")
        stop_listen = st.button("End Listening",type="primary")

        status_placeholder = st.empty()

        if listen:
            st.write("Recording...")
        # Open audio stream
            while not stop_listen:

                audio_data = np.array([], dtype=np.float32)

                for _ in range(int(sampling_rate / chunk * 1.5)):
                    # Read audio chunk from the stream
                    data = stream.read(chunk, exception_on_overflow=False)
                    
                    # Convert byte data to numpy array and normalize
                    chunk_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_data = np.concatenate((audio_data, chunk_data))


                with torch.no_grad():
                    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
                    logits = model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    predicted_label = model.config.id2label[predicted_ids.item()]

                if np.max(np.abs(audio_data)) < 0.005:
                    predicted_label = "NO VOICE DETECTED"

                    
                status_placeholder.success(f"Detected: {predicted_label.upper()} ✅")

        if stop_listen:
            st.write("Recording finished.")
            stream.stop_stream()
            stream.close()
            audio.terminate()
