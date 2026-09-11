// The six monsters on one loop: sum of a float array.
//
// The plain reduction is latency bound: one vaddss every ~4 cycles, because
// s += a[i] is a serial dependency chain the compiler may not reassociate.
// Unrolling with UNROLL independent accumulators breaks the chain into UNROLL chains and
// lets the vectorizer pack them into one register. That is the whole speedup.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if __has_include(<simd>)
#include <simd>
#endif

// ---------------------------------------------------------------- baseline

float sum_plain(const float *a, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

// The same serial chain, spelled with the algorithm: std::accumulate fixes the
// order of the adds, so the compiler may not reassociate this one either.
float sum_accumulate(const float *a, int n) {
    return std::accumulate(a, a + n, 0.0f);
}

// std::reduce permits any order of the adds, so the library may split the
// chain for the compiler. Whether it does is a question for the table.
float sum_reduce(const float *a, int n) {
    return std::reduce(a, a + n, 0.0f);
}

// ------------------------------------------- monster 1: hand unrolled, copy paste

float sum_hand(const float *a, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    for (; i < n; i++) {
        s0 += a[i];
    }
    return (s0 + s1) + (s2 + s3);
}

// ------------------------------------------------- monster 2: macro magic

#define ACC4(i, n, BODY)                \
    for (; i + 3 < n; i += 4) {         \
        BODY(0) BODY(1) BODY(2) BODY(3) \
    }

float sum_macro(const float *a, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    #define BODY(k) s##k += a[i + k];
    ACC4(i, n, BODY)
    #undef BODY
    for (; i < n; i++) {
        s0 += a[i];
    }
    return (s0 + s1) + (s2 + s3);
}

// ---------------------------------------------------- monster 3a: GCC unroll pragma

float sum_pragma_unroll(const float *a, int n) {
    float s = 0;
#pragma GCC unroll 4
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

// ---------------------------------------------------- monster 3b: OpenMP SIMD reduction

float sum_pragma_omp(const float *a, int n) {
    float s = 0;
#pragma omp simd reduction(+:s)
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

// ------------------------------------------------ monster 4: dispatch ladder

template <int UNROLL>
float sum_unrolled(const float *a, int n) {
    auto sums = std::array<float, UNROLL>{};
    int i = 0;
    for (; i + UNROLL <= n; i += UNROLL) {
        for (int k = 0; k < UNROLL; ++k) {
            sums[k] += a[i + k];
        }
    }
    sums[0] = std::reduce(a + i, a + n, sums[0]);
    return std::reduce(sums.begin(), sums.end(), 0.0f);
}

float sum_dispatch(const float *a, int n, int unroll) {
    if (unroll == 2) {
        return sum_unrolled<2>(a, n);
    }
    if (unroll == 4) {
        return sum_unrolled<4>(a, n);
    }
    if (unroll == 8) {
        return sum_unrolled<8>(a, n);
    }
    if (unroll == 16) {
        return sum_unrolled<16>(a, n);
    }
    throw std::invalid_argument("Unsupported unroll factor");
}

// -------------------------------------------------- monster 5: hand-written assembly

#ifdef __AVX2__
// The kernel is sum_asm.s; the assembler pastes it here. rdi = a, rsi = m, m % 128 == 0.
asm(".include \"sum_asm.s\"");
extern "C" float sum_asm_kernel(const float *a, long m);

float sum_asm(const float *a, int n) {
    if (n < 128) { // the kernel is a do-while: one full 128-float iteration, always
        return sum_plain(a, n);
    }
    long m = n & ~127L;
    float r = sum_asm_kernel(a, m);
    for (long i = m; i < n; i++) {
        r += a[i];
    }
    return r;
}
#endif

// ------------------------------- honorable mention: compiler built-in vectors

typedef float v8 __attribute__((vector_size(32), aligned(4)));

float sum_builtin(const float *a, int n) {
    v8 s[16] = {};
    int i = 0;
    for (; i + 127 < n; i += 128) {
        for (int k = 0; k < 16; k++) {
            s[k] += *(v8 *)(a + i + 8 * k);
        }
    }
    const v8 t = std::reduce(s, s + 16, v8{});
    const auto *lanes = reinterpret_cast<const float *>(&t);
    const float r = std::reduce(lanes, lanes + 8, 0.0f);
    return std::reduce(a + i, a + n, r);
}

// ----------------------------------------- C++20: unroll with a pack

template <int UNROLL>
auto sum_pack(std::span<const float> a) -> float {
    auto sums = std::array<float, UNROLL>{};
    const int n = int(a.size());
    int i = 0;
    for (; i + UNROLL - 1 < n; i += UNROLL) {
        [&]<int... K>(std::integer_sequence<int, K...>) {
            ((sums[K] += a[i + K]), ...);
        }(std::make_integer_sequence<int, UNROLL>{});
    }
    sums[0] = std::reduce(a.begin() + i, a.end(), sums[0]);
    return std::reduce(sums.begin(), sums.end(), 0.0f);
}

// ------------------------------- C++26: the pack becomes a statement

#ifdef __cpp_expansion_statements
#include <ranges> // std::views::iota, the range template for walks

// `template for` repeats the body once per element, with `k` a constant expression
// in each copy. No lambda, no integer_sequence, no fold: s[k] is an ordinary index.
template <int UNROLL>
float sum_expand(std::span<const float> a) {
    const int n = int(a.size());
    auto s = std::array<float, UNROLL>{};
    int i = 0;
    for (; i + UNROLL - 1 < n; i += UNROLL) {
        template for (constexpr int k : std::views::iota(0, UNROLL)) {
            s[k] += a[i + k];
        }
    }
    s[0] = std::reduce(a.begin() + i, a.end(), s[0]);
    return std::reduce(s.begin(), s.end(), 0.0f);
}
#endif

// ------------------------------- C++17: the pack needs a helper

template <class F, int... K>
void static_for(F f, std::integer_sequence<int, K...>) {
    (f(std::integral_constant<int, K>{}), ...);
}

template <int UNROLL>
float sum_pack17(const float *a, int n) {
    auto s = std::array<float, UNROLL>{};
    int i = 0;
    for (; i + UNROLL - 1 < n; i += UNROLL) {
        static_for([&](auto k) { s[k] += a[i + k]; },
                   std::make_integer_sequence<int, UNROLL>{});
    }
    s[0] = std::reduce(a + i, a + n, s[0]);
    return std::reduce(s.begin(), s.end(), 0.0f);
}

// one fold expression writes monster 4's ladder

template <int... UNROLL, class F>
bool dispatch(int unroll, F f, std::integer_sequence<int, UNROLL...>) {
    return ((unroll == UNROLL ? (f(std::integral_constant<int, UNROLL>{}), true) : false) || ...);
}

float sum_pack_dispatch(const float *a, int n, int unroll) { // unroll known at run time only
    float r = 0;
    if (!dispatch(unroll, [&](auto k) { r = sum_pack17<k>(a, n); },
                  std::integer_sequence<int, 2, 4, 8, 16, 32>{})) {
        r = sum_plain(a, n); // no kernel for this unroll
    }
    return r;
}

// The policy can change without touching any kernel body.

int vector_register_count() {
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_cpu_supports("avx512f")) {
        return 32;
    }
    return 16;
#elif defined(__aarch64__)
    return 32;
#else
    return 8;
#endif
}

auto choose_unroll() -> int {
    const auto registers = vector_register_count(); // runtime cpuid probe
    return std::max(registers, 4);
}

float sum_pack_best(std::span<const float> a) {
    return sum_pack_dispatch(a.data(), int(a.size()), choose_unroll());
}

// and the ladder of monster 4 is the list it iterates
#ifdef __cpp_expansion_statements
float sum_expand_best(std::span<const float> a) {
    const int unroll = choose_unroll();
    template for (constexpr int UNROLL : {2, 4, 8, 16, 32}) {
        if (unroll == UNROLL) {
            return sum_expand<UNROLL>(a);
        }
    }
    return sum_plain(a.data(), int(a.size())); // no kernel for this factor
}
#endif

// ----------------------------------------- C++26: std::simd

#if (defined(__cpp_lib_simd) || defined(__glibcxx_simd)) && defined(__cpp_expansion_statements)
#include <functional> // std::plus, the reduction std::simd::reduce takes

template <int UNROLL>
float sum_std_simd(std::span<const float> a) {
    namespace simd = std::simd;
    using v = simd::vec<float>;
    static constexpr int width = int(v::size());
    static constexpr int step = UNROLL * width;
    const int n = int(a.size());
    const int vector_end = n - n % step;
    std::array<v, UNROLL> s{};
    for (int i = 0; i < vector_end; i += step) {
        template for (constexpr int k : std::views::iota(0, UNROLL)) {
            s[k] += simd::unchecked_load<v>(a.data() + i + k * width, width);
        }
    }
    const v total = std::reduce(s.begin(), s.end(), v{});
    return std::reduce(a.data() + vector_end, a.data() + n,
                           simd::reduce(total, std::plus<>{}));
}

float sum_std_simd_best(std::span<const float> a) {
    const int unroll = choose_unroll();
    template for (constexpr int UNROLL : {2, 4, 8, 16, 32}) {
        if (unroll == UNROLL) {
            return sum_std_simd<UNROLL>(a);
        }
    }
    return sum_plain(a.data(), int(a.size()));
}
#endif

// ------------------------------------------------------------------ harness

using clk = std::chrono::steady_clock;

template <class F>
double min_time(F f, int reps = 20000) {
    double m = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        const double d = std::chrono::duration<double>(t1 - t0).count();
        if (d < m) {
            m = d;
        }
    }
    return m;
}

using sum_fn = float (*)(std::span<const float>);

template <float (*F)(const float *, int)>
float call_ptr(std::span<const float> a) {
    return F(a.data(), int(a.size()));
}

template <int UNROLL>
float call_dispatch(std::span<const float> a) {
    return sum_dispatch(a.data(), int(a.size()), UNROLL);
}

int main(int argc, char **argv) {
    // 32 KiB fits L1. alignas(64) is not optional: a heap block is not
    // cacheline-aligned, and an unaligned 64-byte load splits a cache line
    // every iteration, which misranks wide-vector kernels.
    constexpr int n = 1 << 13;
    alignas(64) static std::array<float, n> a{};
    // Small integers: every partial sum is exact in float, so reassociation is
    // bit-exact and the check needs no tolerance.
    for (int i = 0; i < n; i++) {
        a[i] = float(i % 7 + 1);
    }
    const float reference = sum_plain(a.data(), n);

    const std::span<const float> input(a);

    // Positive controls for the two guards, which no variant row reaches: 3 has
    // no kernel, so the fold dispatch must fall back and the ladder must throw.
    if (sum_pack_dispatch(a.data(), n, 3) != reference) {
        return 1;
    }
    try {
        sum_dispatch(a.data(), n, 3);
        return 1;
    } catch (const std::invalid_argument &) {
    }

    struct variant {
        const char *name;
        sum_fn f;
    };
    const variant variants[] = {
        {"sum_plain", call_ptr<sum_plain>},
        {"sum_accumulate", call_ptr<sum_accumulate>},
        {"sum_reduce", call_ptr<sum_reduce>},
        {"sum_hand", call_ptr<sum_hand>},
        {"sum_macro", call_ptr<sum_macro>},
        {"sum_pragma_unroll", call_ptr<sum_pragma_unroll>},
        {"sum_pragma_omp", call_ptr<sum_pragma_omp>},
        {"sum_dispatch<16>", call_dispatch<16>},
#ifdef __AVX2__
        {"sum_asm", call_ptr<sum_asm>},
#endif
        {"sum_builtin", call_ptr<sum_builtin>},
        {"sum_pack<8>", sum_pack<8>},
#ifdef __cpp_expansion_statements
        {"sum_expand<8>", sum_expand<8>},
#endif
        {"sum_pack17<8>", call_ptr<sum_pack17<8>>},
        {"sum_pack_best", sum_pack_best},
#ifdef __cpp_expansion_statements
        {"sum_expand_best", sum_expand_best},
#endif
#if (defined(__cpp_lib_simd) || defined(__glibcxx_simd)) && defined(__cpp_expansion_statements)
        {"sum_std_simd_best", sum_std_simd_best},
#endif
    };

    // Variant names on the command line select what runs; no argument runs all.
    // sum_plain always runs: it is the reference the check compares against and
    // the baseline the speedup column divides by.
    const auto selected = [argc, argv](const char *name) {
        if (argc < 2 || std::strcmp(name, "sum_plain") == 0) {
            return true;
        }
        for (int k = 1; k < argc; ++k) {
            if (std::strcmp(argv[k], name) == 0) {
                return true;
            }
        }
        return false;
    };

    // Bound control: every size here is prime, so no unroll factor and no vector
    // width divides it and each kernel runs a partial last group. Each size gets
    // its own heap block, so a read past the end is a wrong answer here and an
    // out-of-bounds report under the sanitizer. The table never reaches this.
    int bad = 0;
    for (const int m : {1, 3, 7, 31, 127, 257, 8191}) {
        const std::vector<float> probe(a.begin(), a.begin() + m);
        const float want = sum_plain(probe.data(), m);
        for (const auto &v : variants) {
            const float got = v.f(probe);
            if (got != want) {
                printf("bound check: %s at n = %d gave %.1f, want %.1f\n", v.name, m, double(got),
                       double(want));
                bad++;
            }
        }
    }
    if (bad != 0) {
        return 1;
    }
    printf("bound check: %zu variants exact at n = 1, 3, 7, 31, 127, 257, 8191\n", std::size(variants));

    const double bytes = double(n) * sizeof(float); // one stream read
    printf("n = %d floats, min of 20000 runs\n\n", n);
    printf("%-20s %10s %10s %8s  %s\n", "variant", "ns", "GB/s", "speedup", "check");
    double t0 = 0;
    for (const auto &v : variants) {
        if (!selected(v.name)) {
            continue;
        }
        const float got = v.f(input);
        const double t = min_time([&] { [[maybe_unused]] volatile float x = v.f(input); });
        if (t0 == 0) {
            t0 = t;
        }
        printf("%-20s %10.0f %10.1f %8.2fx  %s\n", v.name, t * 1e9, bytes / t / 1e9, t0 / t,
               got == reference ? "exact" : "WRONG");
        assert(got == reference);
    }
    return 0;
}
