"""One entry point for refitting or replaying the published prediction evidence."""
from pathlib import Path
import argparse, subprocess, sys

HERE = Path(__file__).resolve().parent

def run(script, *args):
    command = [sys.executable, str(HERE / script), *map(str, args)]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fit', 'replay'])
    parser.add_argument('--archives', type=Path, default=Path.cwd(),
                        help='Directory containing S5_Data.zip, S6_Data.zip and S7_Data.zip')
    args = parser.parse_args()
    if args.mode == 'replay':
        archives = [args.archives / f'S{i}_Data.zip' for i in (5, 6, 7)]
        for path in archives:
            if not path.is_file():
                parser.error(f'Missing prediction archive: {path}')
    run('download_revision_data.py')
    run('revised_pipeline.py', 'prepare')
    run('check_statistics.py')
    if args.mode == 'fit':
        run('revised_pipeline.py', 'classical')
        run('revised_pipeline.py', 'neural')
        run('rebenchmark_inference.py')
    else:
        run('restore_prediction_archives.py', *archives)
    run('recompute_run_metrics.py')
    run('revised_statistics.py')
    run('make_figures.py')
    print('Completed. See revision_20260916/tables and revision_20260916/figures.', flush=True)

if __name__ == '__main__':
    main()
