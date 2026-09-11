// C++17 baseline: poet supplies the compile-time unroll and the runtime
// dispatch, xsimd supplies the portable vector type. The kernels take
// (pointer, count), the spelling that compiles under every standard; the
// std::span entry points sit behind __cpp_lib_span and are the ones the
// correctness check calls when the compiler has them.

#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef __cpp_lib_span
#include <span>
#endif

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>

auto sum_plain(const float *a, int n) -> float {
    auto s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += a[i];
    }
    return s;
}

template <int UNROLL>
auto sum_poet(const float *a, int n) -> float {
    auto s = std::array<float, UNROLL>{};
    poet::dynamic_for<UNROLL>(0, n, [&](auto lane, auto i) { s[lane] += a[i]; });
    return std::reduce(s.begin(), s.end(), 0.0f);
}

using unrolls = std::integer_sequence<int, 2, 4, 8, 16, 32>;

auto sum_poet_dispatch(const float *a, int n, int unroll) -> float {
    return poet::dispatch(poet::throw_on_no_match,
                          [=](auto tag) { return sum_poet<tag>(a, n); },
                          poet::dispatch_param<unrolls>{unroll});
}

// Runtime query on purpose. poet::vector_register_count() answers the same
// question and compiles here (consteval in C++20, constexpr in C++17), but it
// reads -march at compile time, so the compiler would fold choose_unroll() and
// the dispatch with it, and the benchmark would stop measuring a dispatch.
auto vector_register_count() -> int {
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
    const int accumulators = vector_register_count();
    if (accumulators >= 16) {
        return 16;
    }
    if (accumulators >= 8) {
        return 8;
    }
    return 4;
}

auto sum_poet_best(const float *a, int n) -> float {
    return sum_poet_dispatch(a, n, choose_unroll());
}

template <int UNROLL>
auto sum_xsimd(const float *a, int n) -> float {
    using v = xsimd::batch<float, xsimd::best_arch>;
    constexpr std::size_t width = v::size;
    auto s = std::array<v, UNROLL>{};
    const auto block = std::size_t{UNROLL} * width; // power of two, so the mask works
    const auto vector_end = std::size_t(n) & ~(block - 1);
    poet::dynamic_for<UNROLL, width>(std::size_t{0}, vector_end, [&](auto lane, std::size_t i) {
        s[lane] += v::load_unaligned(a + i);
    });

    const v total = std::reduce(s.begin(), s.end(), v(0.0f));
    return std::reduce(a + vector_end, a + n, xsimd::reduce_add(total));
}

auto sum_xsimd_best(const float *a, int n) -> float {
    return poet::dispatch(poet::throw_on_no_match,
                          [=](auto tag) { return sum_xsimd<tag>(a, n); },
                          poet::dispatch_param<unrolls>{choose_unroll()});
}

#ifdef __cpp_lib_span
// C++20 spelling of the same three entry points: one bound argument, no length.
auto sum_plain(std::span<const float> a) -> float { return sum_plain(a.data(), int(a.size())); }

auto sum_poet_best(std::span<const float> a) -> float { return sum_poet_best(a.data(), int(a.size())); }

auto sum_xsimd_best(std::span<const float> a) -> float { return sum_xsimd_best(a.data(), int(a.size())); }
#endif

using clk = std::chrono::steady_clock;

template <class F>
auto min_time(F f, int reps = 20000) -> double {
    double m = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = clk::now();
        f();
        const auto t1 = clk::now();
        const double d = std::chrono::duration<double>(t1 - t0).count();
        if (d < m) {
            m = d;
        }
    }
    return m;
}

using sum_fn = float (*)(const float *, int);

auto main(int argc, char **argv) -> int {
    // 32 KiB fits L1. alignas(64) is not optional: a heap block is not
    // cacheline-aligned, and an unaligned 64-byte load splits a cache line
    // every iteration, which misranks wide-vector kernels.
    constexpr int n = 1 << 13;
    alignas(64) static std::array<float, n> a{};
    for (int i = 0; i < n; ++i) {
        a[i] = float(i % 7 + 1);
    }

    const float *const input = a.data();
#ifdef __cpp_lib_span
    const std::span<const float> checked(a);
    const float reference = sum_plain(checked);
    if (sum_poet_best(checked) != reference || sum_xsimd_best(checked) != reference) {
        return 1;
    }
#else
    const float reference = sum_plain(input, n);
#endif
    // Positive control: every supported factor must reproduce the reference, and an
    // unsupported one must throw instead of silently selecting the wrong kernel.
    for (const int unroll : {2, 4, 8, 16, 32}) {
        if (sum_poet_dispatch(input, n, unroll) != reference) {
            return 1;
        }
    }
    try {
        sum_poet_dispatch(input, n, 3);
        return 1;
    } catch (const std::runtime_error &) {
    }

    struct variant {
        const char *name;
        sum_fn f;
    };
    const variant variants[] = {
        {"sum_plain", sum_plain},
        {"sum_poet<8>", sum_poet<8>},
        {"sum_poet_best", sum_poet_best},
        {"sum_xsimd_best", sum_xsimd_best},
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
            const float got = v.f(probe.data(), m);
            if (got != want) {
                std::printf("bound check: %s at n = %d gave %.1f, want %.1f\n", v.name, m,
                            double(got), double(want));
                ++bad;
            }
        }
    }
    if (bad != 0) {
        return 1;
    }
    std::printf("bound check: %zu variants exact at n = 1, 3, 7, 31, 127, 257, 8191\n",
                std::size(variants));

    const double bytes = double(n) * sizeof(float);
    std::printf("n = %d floats, min of 20000 runs\n\n", n);
    std::printf("%-20s %10s %10s %8s  %s\n", "variant", "ns", "GB/s", "speedup", "check");
    double t0 = 0;
    for (const auto &v : variants) {
        if (!selected(v.name)) {
            continue;
        }
        const float got = v.f(input, n);
        const double t = min_time([&] {
            [[maybe_unused]] volatile float x = v.f(input, n);
        });
        if (t0 == 0) {
            t0 = t;
        }
        std::printf("%-20s %10.0f %10.1f %8.2fx  %s\n",
                    v.name,
                    t * 1e9,
                    bytes / t / 1e9,
                    t0 / t,
                    got == reference ? "exact" : "WRONG");
        assert(got == reference);
    }
    return 0;
}
