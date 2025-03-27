# 🎧 Gender Audio Classification App

  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.8-blue?style=flat-square" /></a>
  <a href="https://streamlit.io/"><img alt="st" src="https://img.shields.io/badge/Made with-Streamlit-blueviolet?style=flat-square" /></a>
  <a href="https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"><img alt="st" src="https://img.shields.io/badge/Wave2Vec-yellow" /></a>
  <a href="https://openai.com/"><img alt="st" src="https://img.shields.io/badge/PyAudio-green" /></a>
  
This repository contains a real-time audio classification app built using Streamlit, PyAudio, and Hugging Face's Wave2Vec2 Transformer model. The app records audio from the microphone, processes it, and predicts the spoken content using a pre-trained speech recognition model.

## 🚀 Features

- ✅ Real-time audio recording using PyAudio
- ✅ Speech classification using a pre-trained Wave2Vec2 Transformer model
- ✅ Live inference & result display in a Streamlit web app

  
## App Layout
  ![alt text](https://github.com/Tejas-Shanbhag/Gender_Audio_Classification/blob/main/assets/app.png)


## 🔧 Dependencies

- streamlit - Web interface for user interaction

- pyaudio - Real-time audio recording

- numpy - Audio data processing

- torch - Deep learning framework

- transformers - Hugging Face library for Wave2Vec2

Install missing dependencies using:
```bash
pip install streamlit pyaudio numpy torch transformers
```

## 🎯 How It Works

1. The user clicks "Start Listening" to begin recording.

2. Audio data is captured from the microphone using PyAudio.

3. The audio signal is preprocessed and normalized.

4. The Wave2Vec2 Transformer model processes the audio and predicts the spoken content.

5. The predicted label is displayed in real-time.


<video src="assets/app_video.mp4" controls width="640" height="360"></video>
