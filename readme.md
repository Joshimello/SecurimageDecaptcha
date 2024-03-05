# SecurimageDecaptcha

Solving the Securimage captcha library by audio recognition using convnets via tensorflow

## Installation (API)

To run the recognition API

```bash
pip install -r requirements.txt
```

Then run the script

```bash
./run.sh
```

Or run manually

```bash
sudo gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5001
```

## Usage (API)

Make a GET request to the api, passing in the url of the audio and (optional) php session id

```
https://example.api:5001?url=AUDIOURL&id=PHPSESSID
```

Or visit the fastAPI docs at

```
https://example.api:5001/docs
```

## Usage (Pre-trained Model)

Two formats of the audio recognition model is available in the /models folder  
These models are for single digit recognition, separation is still required

- Tensorflow saved_model format
- TFLite tflite format

Loading the saved_model format

```py
import tensorflow as tf

model = tf.saved_model.load("saved")
# tf operation #1
prediction = model('./path/to/audio')
# tf operation #2
prediction = model(AUDIO_DATA)
```

Loading the tflite format

```py
import tflite_runtime.interpreter as tflite

model = tflite.Interpreter(model_path='model.tflite')

#or

import tensorflow as tf

model = tf.lite.Interpreter(model_path='model.tflite')


input_details = model.get_input_details()
output_details = model.get_output_details()
model.allocate_tensors()
model.set_tensor(input_details[0]['index'], x)
model.invoke()

prediction = model.get_tensor(output_details[0]['index'])
```

## Usage (Training/Notebook)

The full experimental process and training was done on Google Colab  
The notebook can be found in the /notebooks folder  
Data used for training can be found in the /training folder

## Contributing

Pull requests are always welcomed.

## License

MIT