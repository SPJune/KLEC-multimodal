import glob
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


pattern = "/data2/ai_champion/silent_speech_dataset/*/*/data/audio/*.flac"
flac_paths = sorted(glob.glob(pattern, recursive=True))

print(f"Found {len(flac_paths)} files")

durations = []


for p in flac_paths:
    try:
        info = sf.info(p)
        # frames / samplerate = seconds
        duration = info.frames / info.samplerate
        durations.append(duration)
    except Exception as e:
        print(f"Error reading {p}: {e}")

durations = np.array(durations, dtype=float)

if len(durations) == 0:
    raise RuntimeError("No durations computed. Check the path pattern.")


d_min = float(durations.min())
d_max = float(durations.max())
d_mean = float(durations.mean())
d_median = float(np.median(durations))
d_std = float(durations.std())

print(f"num_files: {len(durations)}")
print(f"min:     {d_min:.4f} sec")
print(f"max:     {d_max:.4f} sec")
print(f"mean:    {d_mean:.4f} sec")
print(f"median:  {d_median:.4f} sec")
print(f"std:     {d_std:.4f} sec")


plt.figure(figsize=(8, 5))
plt.hist(durations, bins=50, edgecolor="black", alpha=0.7)
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.title("FLAC duration histogram")


out_dir = Path("./analysis_plots")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "flac_duration_hist.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Histogram saved to: {out_path}")

