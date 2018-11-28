from PIL import Image
import io
import requests


def encoded_image(image_path):
    image = Image.open(image_path)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    width, height = image.size
    encoded_jpg_io = io.BytesIO()
    image.save(encoded_jpg_io, format='JPEG')
    encoded_jpg = encoded_jpg_io.getvalue()
    return encoded_jpg, width, height


def open_image(path_or_url):
    if path_or_url[:4] == 'http':
        res = requests.get(path_or_url, timeout=10)
        img = Image.open(io.BytesIO(res.content))
    else:
        img = Image.open(path_or_url)
    return img
