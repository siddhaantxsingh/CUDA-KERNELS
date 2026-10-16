// IEEE binary16 (fp16) conversions on raw bits, for host code: the tests,
// the emulator's mma model and the GPU benchmark's reference. Kernels keep
// fp16 data as uint16_t bits and never convert on the device.
#pragma once
#include <cstdint>
#include <cstring>

namespace kc {

// float -> fp16, round to nearest even; overflow -> inf, NaN stays NaN,
// subnormals handled.
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t exp = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (exp == 0xffu) return uint16_t(sign | 0x7c00u | (man ? 0x200u | (man >> 13) : 0u));
    int e = int(exp) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);  // below half the smallest subnormal: rounds to zero
        man |= 0x800000u;                    // implicit bit
        int shift = 14 - e;                  // 14..24
        uint32_t h = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (h & 1u))) h++;
        return uint16_t(sign | h);  // a carry into the exponent field gives the smallest normal: correct
    }
    uint32_t h = (uint32_t(e) << 10) | (man >> 13);
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;  // a carry may round up to inf: correct
    return uint16_t(sign | h);
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t x;
    if (exp == 0x1fu) {
        x = sign | 0x7f800000u | (man << 13);
    } else if (exp == 0) {
        if (man == 0) {
            x = sign;
        } else {  // subnormal: normalise
            int e = -1;
            do {
                man <<= 1;
                e++;
            } while (!(man & 0x400u));
            x = sign | (uint32_t(127 - 15 - e) << 23) | ((man & 0x3ffu) << 13);
        }
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

}  // namespace kc
