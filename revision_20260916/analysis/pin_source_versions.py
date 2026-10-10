"""Verify retained inputs against immutable upstream Git blobs and record provenance.

Release preparation helper; ordinary reproduction uses download_revision_data.py.
No authentication token is used or needed.
"""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/'Healthcare_NLP_CPU_Project/data/raw'
def api(path):
    req=Request('https://api.github.com/'+path,
                headers={'Accept':'application/vnd.github+json','User-Agent':'PONE-D-26-34455-reproducibility'})
    with urlopen(req,timeout=60) as response:return json.load(response)

files=[('jind11/PubMed-PICO-Detection','splitted/PICO_train.txt','pubmed_pico_train.txt'),
       ('jind11/PubMed-PICO-Detection','splitted/PICO_dev.txt','pubmed_pico_dev.txt'),
       ('jind11/PubMed-PICO-Detection','splitted/PICO_test.txt','pubmed_pico_test.txt'),
       ('Franck-Dernoncourt/pubmed-rct','PubMed_20k_RCT/test.txt','pubmed_20k_rct_test.txt')]
commits={repo:api(f'repos/{repo}/commits/master')['sha'] for repo in sorted({x[0] for x in files})}
records=[]
for repo,path,name in files:
    commit=commits[repo];meta=api(f'repos/{repo}/contents/{path}?ref={commit}')
    content=(RAW/name).read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    assert blob==meta['sha'],f'Upstream and retained content differ: {name}'
    records.append({'filename':name,'repository':repo,'upstream_path':path,'commit':commit,
                    'git_blob_sha1':blob,'sha256':hashlib.sha256(content).hexdigest(),
                    'bytes':len(content),'url':f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'})
(ROOT/'data/source_provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
target=ROOT/'analysis/download_revision_data.py';source=target.read_text(encoding='utf-8')
for record in records:
    old=f"https://raw.githubusercontent.com/{record['repository']}/master/{record['upstream_path']}"
    source=source.replace(old,record['url'])
target.write_text(source,encoding='utf-8')
print('PASS: four retained inputs match immutable upstream Git blobs; downloader URLs pinned.')
