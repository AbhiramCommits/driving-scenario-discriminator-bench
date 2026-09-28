#pragma once

#include <cstdint>

namespace dsd {

// SplitMix64: fast deterministic seed mixer used to derive independent per-stream
// seeds from a single scenario id, keeping results reproducible across thread counts.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

} // namespace dsd
