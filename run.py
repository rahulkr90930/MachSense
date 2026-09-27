from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(cmd): subprocess.run(cmd,cwd=ROOT,check=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',nargs='?',choices=['app','paderborn','benchmark','test']); action=ap.parse_args().action
    if action in (None,'app'): return run([sys.executable,'-m','streamlit','run','app/streamlit_app.py'])
    if action=='paderborn': return run([sys.executable,'-m','jupyter','notebook','notebooks/01_Paderborn_End_to_End.ipynb'])
    if action=='benchmark': return run([sys.executable,'-m','jupyter','notebook','notebooks/02_Benchmark_End_to_End.ipynb'])
    if action=='test': return run([sys.executable,'-m','pytest','-q'])
if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MachSense] Stopped.")
        sys.exit(0)
