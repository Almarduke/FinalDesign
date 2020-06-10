# %%

import time
import sys
import json
import re
import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import multiprocessing as mp

# %%

START_PAGE = 1
FINAL_PAGE = 40
ORIGINAL = False
QUERY_SEARCH = None
QUERY_GROUPURL = 'https://www.flickr.com/groups/scenery/pool/'
QUERY_GROUPID = '78249294@N00'
IMG_RESOLUTION_URL = "url_o" if ORIGINAL else "url_l"  # url_c,url_l,url_m,url_n,url_q,url_s,url_sq,url_t,url_z

METADATA_JSON_FOLDER = './download/info'
URL_JSON_FOLDER = './download/url'
IMG_FOLDER = './download/images'

with open('credentials.json') as infile:
    creds = json.load(infile)
    KEY = creds['KEY']
    SECRET = creds['SECRET']

for folder in [METADATA_JSON_FOLDER, URL_JSON_FOLDER, IMG_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


# %%

def get_photo_metadata(page=1):
    params = {
        'content_type': '7',
        'per_page': '500',
        'media': 'photos',
        'format': 'json',
        'advanced': 1,
        'nojsoncallback': 1,
        'extras': f'{IMG_RESOLUTION_URL},o_dims',
        'page': page,
        'api_key': KEY
    }

    if QUERY_SEARCH is not None:
        params['method'] = 'flickr.photos.search',
        params['text'] = QUERY_SEARCH
    elif QUERY_GROUPID is not None:
        params['method'] = 'flickr.groups.pools.getPhotos',
        params['group_id'] = QUERY_GROUPID

    results = requests.get('https://api.flickr.com/services/rest', params=params, headers=headers).json()
    return results


def get_group_id_from_url(url):
    if url is None:
        return None
    params = {
        'method': 'flickr.urls.lookupGroup',
        'url': url,
        'format': 'json',
        'api_key': KEY,
        'nojsoncallback': 1
    }
    results = requests.get('https://api.flickr.com/services/rest', params=params).json()
    return results['group']['id']


# %%

def download_metadata(page):
    file_path = os.path.join(METADATA_JSON_FOLDER, f'page{page}.json')
    if os.path.exists(file_path):
        return

    results = get_photo_metadata(page)
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)
        print(f'Metadata of page {page} dumped')


def save_photo_id_and_urls(page):
    file_path = os.path.join(URL_JSON_FOLDER, f'page{page}.json')
    if os.path.exists(file_path):
        return

    photo_urls = []
    with open(os.path.join(METADATA_JSON_FOLDER, f'page{page}.json'), 'r') as json_file:
        photos = json.load(json_file)['photos']['photo']
        for photo in photos:
            if IMG_RESOLUTION_URL in photo.keys():
                photo_urls.append((photo['id'], photo[IMG_RESOLUTION_URL]))

    with open(file_path, 'w') as json_file:
        json.dump(photo_urls, json_file)
        print(f'Photo urls of page {page} dumped')
        return photo_urls


def get_photo_id_and_urls(page):
    with open(os.path.join(URL_JSON_FOLDER, f'page{page}.json'), 'r') as json_file:
        photo_id_and_urls = json.load(json_file)
        return photo_id_and_urls




def multi_thread_download_image(page):
    photo_id_and_urls = get_photo_id_and_urls(page)
    downloaded_num = 0
    total_num = len(photo_id_and_urls)

    def download_file(a):
        # p_id, p_url = a
        # global downloaded_num
        # extension = p_url.split('.')[-1]
        # filepath = os.path.join(IMG_FOLDER, '{}.{}'.format(p_id, extension))
        print("A")

    #         try:
    #             r = requests.get(url, stream=True)
    #             with open(filepath, 'wb') as f:
    #                 for chunk in r.iter_content(chunk_size=1024):
    #                     if chunk:
    #                         f.write(chunk)
    #             downloaded_num += 1
    #             print(f'"{filename}" downloaded, {downloaded_num} of {total_num}')
    #         except Exception:
    #             pass

    # multi_thread_download_image images
    print('Downloading images')
    pool = mp.Pool(processes=4)
    print(photo_id_and_urls[0])
    for x in photo_id_and_urls:
        pool.apply_async(download_file, args=(x,))
    pool.close()
    pool.join()


# %%

if QUERY_SEARCH is None and QUERY_GROUPURL is None:
    sys.exit('Must specify a search term or group id')

multi_thread_download_image(1)

# %%

# for page in range(START_PAGE, FINAL_PAGE + 1):
#     download_metadata(page)
#     save_photo_id_and_urls(page)
#     multi_thread_download_image(page)

# %%


