/* A `fatbinary` command for clang's CUDA driver, built on NVIDIA's
 * libnvfatbin (PyPI: nvidia-nvfatbin-cu12), for toolkits assembled from
 * wheels, which ship ptxas but not fatbinary. Supports the arguments clang
 * passes: -64 --create OUT --image=profile=sm_XX|compute_XX,file=PATH ... */
#include <nvFatbin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* slurp(const char* path, size_t* n, int nul) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(len + 1);
    if (fread(buf, 1, len, f) != (size_t)len) { perror(path); exit(1); }
    fclose(f);
    buf[len] = 0;
    *n = len + (nul ? 1 : 0);
    return buf;
}

#define CHECK(x) do { nvFatbinResult r = (x); if (r != NVFATBIN_SUCCESS) { fprintf(stderr, "fatbinary: %s: %s\n", #x, nvFatbinGetErrorString(r)); return 1; } } while (0)

int main(int argc, char** argv) {
    const char* out = NULL;
    nvFatbinHandle h;
    const char* opts[] = {"-64"};
    CHECK(nvFatbinCreate(&h, opts, 1));
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--create") && i + 1 < argc) { out = argv[++i]; continue; }
        if (!strncmp(argv[i], "--image=profile=", 16)) {
            char prof[64], file[4096];
            if (sscanf(argv[i] + 16, "%63[^,],file=%4095s", prof, file) != 2) { fprintf(stderr, "fatbinary: bad %s\n", argv[i]); return 1; }
            size_t n;
            if (!strncmp(prof, "sm_", 3)) {
                char* code = slurp(file, &n, 0);
                CHECK(nvFatbinAddCubin(h, code, n, prof + 3, file));
            } else if (!strncmp(prof, "compute_", 8)) {
                char* code = slurp(file, &n, 1);
                CHECK(nvFatbinAddPTX(h, code, n, prof + 8, file, ""));
            } else { fprintf(stderr, "fatbinary: unknown profile %s\n", prof); return 1; }
            continue;
        }
        if (!strcmp(argv[i], "-64") || !strcmp(argv[i], "--64")) continue;
        fprintf(stderr, "fatbinary: ignoring %s\n", argv[i]);
    }
    if (!out) { fprintf(stderr, "fatbinary: --create OUT required\n"); return 1; }
    size_t size;
    CHECK(nvFatbinSize(h, &size));
    void* buf = malloc(size);
    CHECK(nvFatbinGet(h, buf));
    FILE* f = fopen(out, "wb");
    if (!f || fwrite(buf, 1, size, f) != size) { perror(out); return 1; }
    fclose(f);
    CHECK(nvFatbinDestroy(&h));
    return 0;
}
