"""Download exact original public files and require their archived SHA-256 hashes."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

root=Path(__file__).resolve().parents[2]/'Healthcare_NLP_CPU_Project/data/raw'
root.mkdir(parents=True,exist_ok=True)
specs=[
('pubmed_pico_train.txt','https://raw.githubusercontent.com/jind11/PubMed-PICO-Detection/6eecb78707eb3a578242f4128d803841e41e251c/splitted/PICO_train.txt','411836321504fd1df3e01ecba2fe6b66e5a7382ca9f94d6dd69ecf13cb7add43'),
('pubmed_pico_dev.txt','https://raw.githubusercontent.com/jind11/PubMed-PICO-Detection/6eecb78707eb3a578242f4128d803841e41e251c/splitted/PICO_dev.txt','6497c31124004bdf1cd3c7f090eda8274bef0a6e5ca96dcdbd393bc1e0b435b3'),
('pubmed_pico_test.txt','https://raw.githubusercontent.com/jind11/PubMed-PICO-Detection/6eecb78707eb3a578242f4128d803841e41e251c/splitted/PICO_test.txt','05939c46541790f78d580233b3ceb8019c5d67d5a15100d283f02cfbe2312e21'),
('pubmed_20k_rct_test.txt','https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/17ed2cb0590decfca0266add0c76f254f67232b4/PubMed_20k_RCT/test.txt','8511b832c1ee66b3df52df23c16ad81a69711ddd5b3a485421da3f8e9d5b44f4')]
for name,url,expected in specs:
    p=root/name
    content=p.read_bytes() if p.exists() else urlopen(url,timeout=120).read()
    actual=hashlib.sha256(content).hexdigest()
    if actual!=expected:raise RuntimeError(f'Input version changed: {name}: {actual}, expected {expected}. Do not silently substitute.')
    if not p.exists():p.write_bytes(content)
    print(name,actual,flush=True)
