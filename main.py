from time import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import noisereduce as nr
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf
import io
import requests
import tflite_runtime.interpreter as tflite

app = FastAPI()
model = tflite.Interpreter(model_path='model.tflite', num_threads=4)

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filter_segments(y, sr, segments, threshold):
    filtered_segments = []
    for segment in segments:

        duration = (segment[1] - segment[0]) / sr
        if duration > 0.85:
            mid_point = (segment[0] + segment[1]) // 2
            segment1 = (segment[0], mid_point)
            segment2 = (mid_point, segment[1])

            max_amplitude1 = np.max(np.abs(y[segment1[0]:segment1[1]]))
            if max_amplitude1 > threshold:
                filtered_segments.append(segment1)

            max_amplitude2 = np.max(np.abs(y[segment2[0]:segment2[1]]))
            if max_amplitude2 > threshold:
                filtered_segments.append(segment2)
        else:
            max_amplitude = np.max(np.abs(y[segment[0]:segment[1]]))
            if max_amplitude > threshold:
                filtered_segments.append(segment)

    return filtered_segments

def separate_audio_url(url, cookies=None):
    response = requests.get(url, cookies=cookies)
    audio_data = response.content
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
    oy = np.copy(y)

    y = nr.reduce_noise(y=y, sr=sr)
    y = butter_bandpass_filter(y, 200, 300, sr)
    y = np.where(np.abs(y) < 0.01, 0, y)

    segments = librosa.effects.split(y, hop_length=1024)
    segments = filter_segments(y, sr, segments, 0.025)

    segments_audio = []
    for segment in segments:
        segments_audio.append(oy[segment[0]:segment[1]])

    return segments_audio, sr, segments

def preprocess_audio(x):
    desired_length = 9922
    if x.shape[0] < desired_length:
        padding = desired_length - x.shape[0]
        x = np.pad(x, pad_width=((0, padding)), mode='constant')
    elif x.shape[0] > desired_length:
        x = x[:desired_length]
    x = np.reshape(x, newshape=(1, -1))
    return x

@app.get('/', response_class=PlainTextResponse)
async def uwu(url, id=None):
    Y, sr, _ = separate_audio_url(url, {'PHPSESSID': id})

    answer = []
    for audio in Y:
        x = np.asarray(audio, dtype=np.float32)
        x = preprocess_audio(x)
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.allocate_tensors()
        model.set_tensor(input_details[0]['index'], x)
        model.invoke()
        prediction = model.get_tensor(output_details[0]['index'])
        answer.append(np.argmax(prediction[0]))
    
    answer = [str(a) for a in answer if a != 10]
    answer_string = ''.join(answer)
  
    return answer_string