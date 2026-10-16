"""Parse ptxas -v output (tools/device_build.sh) into per-kernel resources and
theoretical occupancy. Usage: python3 tools/resources.py build/device > resources.json"""
import json
import re
import subprocess
import sys

# per-SM limits (CUDA C Programming Guide, compute capability tables)
LIMITS = {
    "sm_80": dict(regs=65536, threads=2048, blocks=32, smem=167936, reserved_smem=1024, name="A100 (sm_80)"),
    # compute capability 8.9 (Ada, e.g. RTX 4060): 1536 threads, 24 blocks, 100 KB shared memory per SM
    "sm_89": dict(regs=65536, threads=1536, blocks=24, smem=102400, reserved_smem=1024, name="RTX 40xx / L4 (sm_89)"),
    "sm_90": dict(regs=65536, threads=2048, blocks=32, smem=233472, reserved_smem=1024, name="H100 (sm_90)"),
}
# threads per block used by the launchers in include/kc/*.cuh
BLOCK = {
    "reduce_atomic_kernel": 256, "reduce_smem_kernel": 256, "reduce_partials_kernel": 256, "reduce_final_kernel": 256,
    "transpose_naive_kernel": 256, "transpose_tiled_kernel": 256, "sgemm_naive_kernel": 256, "sgemm_smem_kernel": 1024,
    "sgemm_reg_kernel": 256, "softmax_warp_kernel": 128, "layernorm_warp_kernel": 128, "flash_attn_kernel": 64,
    "attn_naive_kernel": 64, "hgemm_mma_kernel": 256, "mma_probe_kernel": 32, "softmax_bwd_warp_kernel": 128,
    "layernorm_bwd_rows_kernel": 128, "layernorm_bwd_cols_kernel": 256,
}


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")
    return out[: len(names)]


def occupancy(arch, regs, smem, threads):
    L = LIMITS[arch]
    warps = -(-threads // 32)
    regs_per_warp = -(-max(regs, 1) * 32 // 256) * 256
    by_regs = L["regs"] // (regs_per_warp * warps)
    by_smem = L["smem"] // (smem + L["reserved_smem"]) if smem else L["blocks"]
    by_threads = L["threads"] // (warps * 32)
    blocks = min(by_regs, by_smem, by_threads, L["blocks"])
    # ties go to the hardware thread/block limits: those bind regardless of the kernel
    limiter = min([("threads", by_threads), ("blocks", L["blocks"]), ("registers", by_regs), ("shared memory", by_smem)], key=lambda x: x[1])[0]
    return blocks, blocks * warps * 32 / L["threads"], limiter


def main(d):
    rows = []
    for arch in LIMITS:
        text = open(f"{d}/ptxas_{arch}.txt").read()
        entries = re.split(r"Compiling entry function '", text)[1:]
        mangled = [e.split("'")[0] for e in entries]
        for m, e, name in zip(mangled, entries, demangle(mangled)):
            regs = int(re.search(r"Used (\d+) registers", e).group(1))
            smem = int(m.group(1)) if (m := re.search(r"(\d+) bytes smem", e)) else 0
            stack = int(re.search(r"(\d+) bytes stack frame", e).group(1))
            spill = int(re.search(r"(\d+) bytes spill stores", e).group(1))
            base = name.split("(")[0].split("<")[0].replace("kc::", "").replace("void ", "")
            threads = BLOCK[base]
            blocks, occ, limiter = occupancy(arch, regs, smem, threads)
            rows.append(dict(arch=arch, kernel=name.replace("void ", "").split("(")[0], registers=regs, smem_bytes=smem,
                             stack_bytes=stack, spill_bytes=spill, threads_per_block=threads, blocks_per_sm=blocks,
                             occupancy=round(occ, 4), limiter=limiter))
    json.dump(rows, sys.stdout, indent=1)


main(sys.argv[1])
