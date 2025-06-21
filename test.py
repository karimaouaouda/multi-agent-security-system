import io

import requests
import PIL.Image as Image


response = requests.get('http://localhost:8000/api/download')

image = response.content

img = Image.open(io.BytesIO(image))

img.show()
