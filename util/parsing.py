import os
from collections import defaultdict
import regex as re
import nltk
nltk.download('all')

TRAIN_PATH = "./train"
LANGS = ['bg', 'cs', 'pl', 'ru']
ENTITY_TAGS = ['EVT', 'GPE', 'LOC', 'ORG', 'PER', 'PRO']
# RAW_PATH = os.path.join(TRAIN_PATH, 'raw')
# ANNOTATED_PATH = os.path.join(TRAIN_PATH, 'annotated')

def get_doc_pairs(path, langs=LANGS):
    res = defaultdict(list)
    annotated_path = os.path.join(path, 'annotated')
    raw_path = os.path.join(path, 'raw')
    for lang in LANGS:
        annotated_dir = os.path.join(annotated_path, lang)
        raw_dir = os.path.join(raw_path, lang)
        annotated = sorted(os.listdir(annotated_dir))
        raw = sorted(os.listdir(raw_dir))
        res[lang] = list(map(
                        lambda x: (
                            os.path.join(raw_dir, x[0]),
                            os.path.join(annotated_dir, x[1])
                        ),
                        zip(raw, annotated)))
    return res

def get_entity_spans(document, query, additional_data=[]):
    def find_all(doc, q):
        res = []
        cur_pos = doc.find(q)
        while cur_pos >= 0:
            res.append((cur_pos, cur_pos + len(query)))
            cur_pos = doc.find(q, cur_pos + 1)
        return res

    document = document.lower()
    query = query.lower()
    matches = find_all(document, query)
    if not len(matches) > 0:
        try:
            matches = list(map(
                        lambda x: x.span(),
                        re.finditer('(%s){e<=1}' % query, document)
                        ))
        except:
            matches = []
    matches = list(map(
        lambda x: (x, additional_data),
        matches
    ))
    return matches

def fix_doc_pair(raw_path, annot_path, verbose=False):
    unwanted_chars = r'[\u200b]'
    def clean_string(s):
        return re.sub(unwanted_chars, '', s)\
                .replace('“', '"')\
                .replace('„', '"')
    raw = open(raw_path, 'r')
    annotation = open(annot_path, 'r')

    raw.readline() # part
    raw.readline() # lang
    raw.readline() # date
    raw.readline() # source

    raw_text = clean_string(raw.read())
    annotation.readline() # first line does not contain tags
    total = 0
    errors = 0
    spans = []
    for line in annotation:
        ent_data = line.strip().split('\t')
        ent = clean_string(ent_data[0])
        ent_data = {k: v for k, v in zip(('lemmatized_ent', 'class', 'id'), ent_data[1:])}
        total += 1
        #if raw_text.find(ent) < 0:
        matches = get_entity_spans(raw_text, ent, ent_data)
        spans.extend(matches)
        if not len(matches) > 0:
            if verbose:
                print("Error in doc %s with content:" % raw_path.split('/')[-1])
                print(raw_text)
                print("Couldn't find entity:")
                print(line)
                print('---------------------------')
            errors += 1
        elif len(matches) > 1:
            if verbose:
                print("An entity matched %d times!" % len(matches), ent)
    res = {
        'entity_spans': sorted(spans, key=lambda x: x[0]),
        'text': raw_text,
    }
    return res, (total, errors)

def get_formatted_dataset(path='./train', langs=LANGS, dont_keep_id=True):
    def intersect(s1, e1, s2, e2):
        if e1 <= s2:
            return -1 #left
        elif e2 <= s1:
            return 1 #right
        else:
            return 0 #intersect

    doc_pairs = get_doc_pairs(path, langs)
    docs_and_spans = defaultdict(list)
    final_dataset = defaultdict(list)
    for lang in langs:
        tokenizer = nltk.tokenize.WordPunctTokenizer()
        print(lang)

        for doc_pair in doc_pairs[lang]:
            res, _ = fix_doc_pair(*doc_pair)
            doc_token_spans = list(tokenizer.span_tokenize(res['text']))
            res['token_spans'] = doc_token_spans
            docs_and_spans[lang].append(res)

        for doc in docs_and_spans[lang]:
            token_spans = doc['token_spans']
            entity_spans = doc['entity_spans']
            text = doc['text']
            entity_iter = iter(entity_spans)
            (cur_entity_s, cur_entity_e), ent_data = next(entity_iter)
            ent_tag = 'O'
            new_ent = True
            res_sent = []
            for token_s, token_e in token_spans:
                token = text[token_s:token_e]
                intersection = intersect(token_s, token_e, cur_entity_s, cur_entity_e)
                if intersection == 0:
                    if new_ent:
                        ent_tag = 'B_' + (ent_data['class'] if dont_keep_id else ent_data['id'])
                        new_ent = False
                    else:
                        ent_tag = 'I_' + (ent_data['class'] if dont_keep_id else ent_data['id'])
                elif intersection == 1:
                    new_ent = True
                    ent_tag = '0'
                    try:
                        (cur_entity_s, cur_entity_e), ent_data = next(entity_iter)
                    except StopIteration:
                        cur_entity_s, cur_entity_e = -1, -1
                else:
                    ent_tag = '0'
                    new_ent = True
                res_sent.append((token, ent_tag))
            final_dataset[lang].append(res_sent)
    return final_dataset
