import os
import requests
import re
import zipfile

from tqdm import tqdm
from os.path import dirname, abspath, join, exists

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
    url = "https://drive.google.com/file/d/1d5bOxxJ3ZcMP3zXxwI685T5_Cawhl0V1/view?usp=sharing"
    
    dirname = dirname(abspath(__file__))
    destination = join(dirname, '.', "lfw.zip")
    
    download_from_google_drive(url, destination)       
    unzip(destination, dirname)
    
    
    