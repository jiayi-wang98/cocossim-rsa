"""Run the 24 TPUv3 validation workloads of ae_plots/experiments.py and compare with the measured latencies.

Usage: python3 run_val.py OUT_TAG [extra perf_model flags...]
Each workload runs in its own directory (DRAMSim3 writes its stats files into the cwd), at most 8 at a time.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build/perf_model"
DRAM = ROOT / "dramsim3/configs/HBM2_8Gb_x128.ini"  # override with -dram_cfg in the extra flags
BASE = ["-sa_sz", "128", "-vu_sz", "1024", "-ws", "1", "-f", "0.94"]
CLOCK_MHZ = 940

# Same order as tpu_data in ae_plots/plot_parse.py (microseconds on TPUv3)
TPU_US = [92.8, 74, 86, 41.6, 944, 55, 42.4, 2.2, 238, 142, 168, 89, 665, 43, 4.022, 1.08,
          515.2, 324.4, 327.2, 203.6, 2536, 255, 161, 82]

def resnet(m, k, n, g, w):
    return [f"LayerNorm 32 {m} {w}", f"Conv {m} {k} {n}", f"Activation {m} {w}",
            f"LayerNorm 32 {m} {w}", f"Conv {m} {k} {n}", f"Activation {m} {w}"]

WORKLOADS = (
    [("Matmul", d, [f"Matmul {d[0]} {d[1]} {d[2]}"]) for d in [(4096, 320, 320), (1024, 640, 640), (256, 1280, 1280), (64, 1280, 1280)]]
    + [("DotProduct", d, ["Matmul " + " ".join(map(str, d))]) for d in [(8, 4096, 40, 4096), (8, 1024, 80, 1024), (8, 256, 160, 256), (8, 64, 160, 64)]]
    + [("Conv", d, [f"Conv {d[0]} {d[1]} {d[2]}"]) for d in [(4096, 2880, 320), (1024, 5760, 640), (256, 11520, 1280), (64, 11520, 1280)]]
    + [("Softmax", d, [f"Softmax {d[0]} {d[1]}"]) for d in [(8, 4096), (8, 1024), (8, 256), (8, 64)]]
    + [("ResNet", d, resnet(*d)) for d in [(4096, 2880, 320, 32, 320), (1024, 5760, 640, 32, 640), (256, 11520, 1280, 32, 1280), (64, 11520, 1280, 32, 1280)]]
    + [("SelfAttention", d, ["SelfAttention " + " ".join(map(str, d))]) for d in [(8, 4096, 320, 320), (8, 1024, 640, 640), (8, 256, 1280, 1280), (8, 64, 1280, 1280)]]
)
# The AE script ran ResNet blocks with -c 4 and everything else with -c 1; callers pass -c themselves.


def run(i, tag, extra):
    name, dims, lines = WORKLOADS[i]
    d = ROOT / "val" / tag / f"{i:02d}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "in.txt").write_text("\n".join(lines) + "\n")
    p = subprocess.run([str(BIN), *BASE, "-dram_cfg", str(DRAM), *extra, "-i", "in.txt", "-o", "out.txt"], cwd=d,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=7200)
    if p.returncode != 0:
        return dict(i=i, name=name, dims=dims, error=p.returncode)
    out = (d / "out.txt").read_text().split("\n")
    cycles = int(out[0].split()[1])
    units = [(l.split()[0], float(l.split()[1])) for l in out[1:] if l.strip()]
    return dict(i=i, name=name, dims=dims, cycles=cycles, us=cycles / CLOCK_MHZ, units=units)


if __name__ == "__main__":
    tag, extra = sys.argv[1], sys.argv[2:]
    # runs happen in per-workload directories, so make a -dram_cfg path absolute here
    extra = [str(Path(a).resolve()) if i > 0 and extra[i - 1] == "-dram_cfg" else a for i, a in enumerate(extra)]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: run(i, tag, extra), range(len(WORKLOADS))))
    errs = []
    print(f"{'workload':14s} {'dims':26s} {'sim us':>9s} {'TPU us':>8s} {'err %':>7s}")
    for r in res:
        t = TPU_US[r["i"]]
        if "error" in r:
            print(f"{r['name']:14s} {str(r['dims']):26s} {'ERROR ' + str(r['error']):>9s} {t:8.1f}")
            continue
        e = 100 * (r["us"] - t) / t
        r["tpu_us"], r["err_pct"] = t, e
        errs.append(abs(e))
        print(f"{r['name']:14s} {str(r['dims']):26s} {r['us']:9.1f} {t:8.1f} {e:+7.1f}")
    if errs:
        print(f"mean |error| over {len(errs)} workloads: {sum(errs) / len(errs):.1f}%")
    (ROOT / "val" / f"{tag}.json").write_text(json.dumps(res, indent=1))
