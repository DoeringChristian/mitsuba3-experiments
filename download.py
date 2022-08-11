import requests
import zipfile

url = "https://rgl.s3.eu-central-1.amazonaws.com/scenes/tutorials/scenes.zip"

req = requests.get(url)

filename = url.split('/')[-1]

with open(filename, 'wb') as out_file:
    out_file.write(req.content)


with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall(''.join(map(str, filename.split('.')[:-1])))
