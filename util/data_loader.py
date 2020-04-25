from zipfile import ZipFile
import requests
import os
from helper_functions import is_notebook

if is_notebook():
    from tqdm.tqdm_notebook import tqdm
else:
    from tqdm import tqdm

WORD2VEC_URLS = { # source http://vectors.nlpl.eu/repository/
    'bg': 'http://vectors.nlpl.eu/repository/20/33.zip',
    'ru': 'http://vectors.nlpl.eu/repository/20/65.zip',
    'pl': 'http://vectors.nlpl.eu/repository/20/62.zip',
    'cz': 'http://vectors.nlpl.eu/repository/20/37.zip',
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

def get_language_embeddings(dest='./', force=False):
    for lang, url in WORD2VEC_URLS.items():
        lang_dest = os.path.join(dest, '%s_w2v' % lang)
        if os.path.exists(lang_dest) and not force:
            print("%s is already downloaded, moving on" % lang)
            continue
        print("loading and unzipping wordembeddings for %s" % lang)
        print("result will be located in %s" % lang_dest)
        zip_file = './%s.zip' % lang
        download(url, zip_file)
        with ZipFile(zip_file) as w2v_zip:
            w2v_zip.extractall(os.path.join(dest, '%s_w2v' % lang))
        os.remove(zip_file)

