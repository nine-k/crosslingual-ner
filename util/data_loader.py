from zipfile import ZipFile
import requests

import os
import shutil

from .helper_functions import is_notebook
if is_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

WORD2VEC_URLS = { # source http://vectors.nlpl.eu/repository/
    'bg': 'http://vectors.nlpl.eu/repository/20/33.zip',
    'ru': 'http://vectors.nlpl.eu/repository/20/65.zip',
    'pl': 'http://vectors.nlpl.eu/repository/20/62.zip',
    'cs': 'http://vectors.nlpl.eu/repository/20/37.zip',
}

def download(url, dest):
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dest, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

def download_and_unzip(url, dest):
    zip_file = dest + '.zip'
    download(url, zip_file)
    with ZipFile(zip_file) as f:
        f.extractall(dest)
    os.remove(zip_file)

def download_dataset(url="http://bsnlp.cs.helsinki.fi/TRAININGDATA_BSNLP_2019_shared_task.zip", dest='./train'):
    download_and_unzip(url, dest)
    files = os.listdir(os.path.join(dest, "training_pl_cs_ru_bg_rc1"))
    for f in files:
        shutil.move(os.path.join(dest, "training_pl_cs_ru_bg_rc1", f), os.path.join(dest, f))
    os.rmdir(os.path.join(dest, "training_pl_cs_ru_bg_rc1"))

def download_test(url="http://bsnlp.cs.helsinki.fi/TESTDATA_BSNLP_2019_shared_task.zip", dest='./test'):
    download_and_unzip(url, dest)

def get_language_embeddings(word2vec_urls=WORD2VEC_URLS, dest='./', force=False):
    for lang, url in word2vec_urls.items():
        lang_dest = os.path.join(dest, '%s_w2v' % lang)
        if os.path.exists(lang_dest) and not force:
            print("%s is already downloaded, moving on" % lang)
            continue
        print("loading and unzipping wordembeddings for %s" % lang)
        print("result will be located in %s" % lang_dest)
        download_and_unzip(url, lang_dest)
