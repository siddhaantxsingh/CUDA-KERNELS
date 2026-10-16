"""Disassemble the device cubins (cuobjdump -sass from the CUDA PyPI wheels)
and count tensor-core and related instructions per kernel. Fails if a
tensor-core kernel has no HMMA instruction for some architecture.
Usage: python3 tools/sass_check.py build/device sm_80 sm_89 sm_90 > results/sass.json"""
import json
import re
import subprocess
import sys

d, archs = sys.argv[1], sys.argv[2:]
cuobjdump = "build/cuda-root/bin/cuobjdump"
OPS = ["HMMA", "LDSM", "LDG.E.128", "STS.128", "BAR.SYNC", "FFMA"]
TENSOR_KERNELS = ("hgemm_mma_kernel", "mma_probe_kernel")
rows, failed = [], False
for arch in archs:
    sass = subprocess.run([cuobjdump, "-sass", f"{d}/kernels_{arch}.cubin"], capture_output=True, text=True, check=True).stdout
    open(f"{d}/sass_{arch}.txt", "w").write(sass)
    funcs = re.split(r"\n\s*Function : ", sass)[1:]
    names = [f.split("\n", 1)[0].strip() for f in funcs]
    demangled = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")
    for name, body in zip(demangled, funcs):
        instrs = re.findall(r"/\*[0-9a-f]{4,}\*/\s+([^;]+);", body)
        opcodes = [re.sub(r"^@!?U?P[T0-9]+\s+", "", i.strip()).split()[0] for i in instrs]
        counts = {op: sum(1 for o in opcodes if o == op or o.startswith(op + ".")) for op in OPS}
        hmma_forms = sorted({o for o in opcodes if o.startswith("HMMA")})
        short = name.replace("void ", "").split("(")[0]
        rows.append(dict(arch=arch, kernel=short, instructions=len(instrs), hmma_forms=hmma_forms, **{k.lower().replace(".", "_"): v for k, v in counts.items()}))
        if any(t in short for t in TENSOR_KERNELS) and counts["HMMA"] == 0:
            print(f"FAIL: {arch} {short} has no HMMA", file=sys.stderr)
            failed = True
json.dump(rows, sys.stdout, indent=1)
for r in rows:
    if r["hmma"]:
        print(f"{r['arch']}: {r['kernel']}: {r['hmma']} HMMA {r['hmma_forms']}, {r['ldsm']} LDSM", file=sys.stderr)
sys.exit(1 if failed else 0)
