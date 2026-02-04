import os, glob, subprocess, sys

root = r"C:\Users\Administrator\Desktop"
# locate repo even if path contains non-ascii
matches = glob.glob(os.path.join(root, "**", "temporal-reasoning-benchmark"), recursive=True)
if not matches:
    print("[ERR] temporal-reasoning-benchmark not found under Desktop")
    sys.exit(2)
repo = matches[0]
print("RUN_IN", repo)

cmd = [sys.executable, "-m", "src.runner", "--config", "configs\\pilot_tram_cot0.yaml"]
print("CMD", " ".join(cmd))

ret = subprocess.run(cmd, cwd=repo)
sys.exit(ret.returncode)


