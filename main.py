# Imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys

# Load the model
filepath = './model/trained-mnist-model'
model = load_model(filepath, compile = True)

# Get the input shape for the model layer
input_shape = model.layers[0].input_shape

# Define the FastAPI app
app = FastAPI()

# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction_probabilities: List[float] = []
  identified_number: int


# Define the main route
@app.get('/')
def root_route():
  return { 'error': 'Use GET /prediction instead of the root route!' }


# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)
async def prediction_route(digi_image: UploadFile = File(...)):

  # Ensure that this is an image
  if digi_image.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{digi_image.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await digi_image.read()
    pil_image = Image.open(io.BytesIO(contents))

    # Resize image to expected input shape
    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    # Convert from RGBA to RGB *to avoid alpha channels*
    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    # Convert image into grayscale *if expected*
    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    # Convert image into numpy format
    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    # Scale data (depending on your model)
    numpy_image = numpy_image / 255

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {
      'filename': digi_image.filename,
      'contenttype': digi_image.content_type,
      'prediction_probabilities': prediction.tolist(),
      'identified_number': likely_class
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))

