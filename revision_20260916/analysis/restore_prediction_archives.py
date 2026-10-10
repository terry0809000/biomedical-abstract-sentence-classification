"""Restore precise run CSVs from S5/S6/S7 Data for metric/figure reproduction."""
from pathlib import Path
from evidence_io import read_csv
import io,sys,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
for filename in sys.argv[1:]:
    with zipfile.ZipFile(filename) as z:
        for metadata in [n for n in z.namelist() if n.endswith('/sentences.csv')]:
            split=metadata.split('/')[0];base=read_csv(z.open(metadata),dtype={'abstract_id':str}).drop(columns='row_index')
            for name in [n for n in z.namelist() if n.startswith(split+'/') and n.endswith('.npz')]:
                run=Path(name).stem;arr=np.load(io.BytesIO(z.read(name)));d=base.copy()
                for key in ['raw_score','probability','prediction']:d[key]=arr[key]
                target=ROOT/'runs'/run;target.mkdir(exist_ok=True,parents=True);d.to_csv(target/f'predictions_{split}.csv',index=False)
                print(run,split,len(d))
for run in (ROOT/'runs').iterdir():
    if all((run/f'predictions_{split}.csv').exists() for split in ['selection','calibration','threshold','test','external']):
        (run/'COMPLETE').write_text('Restored from supplied prediction archives; model refitting was not performed.\n')
