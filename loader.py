import os.path
import requests


default_file_url = 'https://drive.google.com/file/d/1H9BXcQL63zfVTtJKw1figeB9SbK43qLP/view?usp=sharing'
default_destination = 'model_state_dict.pth'


def download_file_from_google_drive(url, destination):
    session = requests.Session()

    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def get_state_model(file_url=None, destination=None, download_force=False):

    if file_url is None:
        file_url = default_file_url

    if destination is None:
        destination = default_destination

    if download_force or not os.path.exists(default_destination):
        download_file_from_google_drive(file_url, destination)



