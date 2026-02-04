import argparse
import os
import sys
import glob
import subprocess
from pathlib import Path


def find_repo() -> str:
    # Try current working directory then Desktop recursive
    cwd = Path.cwd()
    if (cwd / 'src').exists() and (cwd / 'configs').exists():
        return str(cwd)

    desktop = Path.home() / 'Desktop'
    matches = glob.glob(str(desktop / '**' / 'temporal-reasoning-benchmark'), recursive=True)
    if matches:
        return matches[0]

    raise SystemExit('[ERR] Cannot locate temporal-reasoning-benchmark. Please run this script from inside the repo.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Config path inside repo, e.g. configs/pilot_tram_sp0.yaml')
    args = ap.parse_args()

    repo = find_repo()
    print('RUN_IN', repo)
    cmd = [sys.executable, '-m', 'src.runner', '--config', args.config]
    print('CMD', ' '.join(cmd))
    subprocess.run(cmd, cwd=repo, check=False)


if __name__ == '__main__':
    main()


