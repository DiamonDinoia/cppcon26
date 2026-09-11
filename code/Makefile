CXX ?= g++
MAKEFLAGS += --no-print-directory
OPTFLAGS ?= -O3 -march=native -fopenmp-simd
WARNFLAGS ?= -Wall -Wextra -pedantic
CODE_CXXSTD ?= -std=c++20
STANDARD_CXXSTD ?= -std=c++17
CPPFLAGS := -isystem deps/poet/include -isystem deps/xsimd/include
TASKSET_CPUS ?= 3,4 # both SMT siblings of one P-core, or the sibling steals FP ports

POET_URL ?= https://github.com/flatironinstitute/poet.git
XSIMD_URL ?= https://github.com/xtensor-stack/xsimd.git

# Variants to run; empty runs all of them. Single-quote any name holding <>,
# or the shell reads it as a redirection.
VARIANTS ?=

.PHONY: all deps run clean

all: code standard

deps: deps/poet/.git deps/xsimd/.git deps/poet/include/poet/version.hpp

deps/poet/.git:
	git clone --depth 1 $(POET_URL) deps/poet

deps/xsimd/.git:
	git clone --depth 1 $(XSIMD_URL) deps/xsimd

deps/poet/include/poet/version.hpp: deps/poet/.git
	cmake -S deps/poet -B deps/poet/build

code: code.cpp sum_asm.s
	$(CXX) $(CODE_CXXSTD) $(OPTFLAGS) $(WARNFLAGS) $< -o $@

standard: standard.cpp deps
	$(CXX) $(STANDARD_CXXSTD) $(OPTFLAGS) $(WARNFLAGS) $(CPPFLAGS) $< -o $@

# Pinned: an unpinned run of either binary measures the scheduler.
run: all
	taskset -c $(TASKSET_CPUS) ./code $(VARIANTS)
	taskset -c $(TASKSET_CPUS) ./standard $(VARIANTS)

clean:
	$(RM) code standard
