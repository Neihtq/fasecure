import re
import pathlib
import requests
import zipfile
import tarfile 
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download zip from Google Drive')
parser.add_argument('--url', type=str)
args = parser.parse_args()

def download_data(url):
    req = requests.get(url, allow_redirects=True)
    open("data.tgz", 'wb').write(req.content)

    with tarfile.open('data.tgz', 'r') as f:
        f.extractall('data')
        f.close()


def download_from_google_drive(drive_url, destination):
    ''' downloads data from google drive into the destination folder
    file of given id must be accessible through shareable link
    
    fails with large data
    '''
    
    URL = re.sub(r"https://drive\.google\.com/file/d/(.*?)/.*?\?usp=sharing", r"https://drive.google.com/uc?export=download&id=\1", drive_url)

    session = requests.Session()
    res = session.get(URL, stream=True)
    token = get_token(res)
    
    if token:
        params = {'id' : id, 'confirm': token}
        res = session.get(URL, params=params, stream=True)
        
    CHUNK_SIZE = 32768 #?
    with open(destination, "wb") as f:
        for chunk in tqdm(res.iter_content(CHUNK_SIZE)):
            if chunk:
                f.write(chunk)
    
    
def get_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
       
    return None


def unzip(path, destination):
    with zipfile.ZipFile(path, 'r') as zf:
        zf.extractall(destination)


if __name__ == "__main__":
    url = args.url
    gdrive = args.gdrive
    
    image_path = pathlib.Path('./data/images')
    destination = pathlib.Path('./data/lfw.zip')
    
    image_path.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    download_from_google_drive(url, destination)       
    unzip(destination, image_path)