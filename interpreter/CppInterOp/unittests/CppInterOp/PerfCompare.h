// gtest-facing relative-performance assertions over Google Benchmark.
//
// One BENCHMARK per dispatch tier; one TEST per ordering claim. The
// first macro use runs all registered benchmarks once (auto-iterated,
// median-quantile), caches the results by name through a custom
// reporter, and later uses hit the cache. Assertions are relative
// (X-not-slower-than-Y), which is stable across hardware where
// absolute ns are not.

#ifndef CPPINTEROP_TEST_PERFCOMPARE_H
#define CPPINTEROP_TEST_PERFCOMPARE_H

#include <benchmark/benchmark.h>
#include "gtest/gtest.h"

#include <map>
#include <string>
#include <vector>

namespace PerfCompare {

inline const std::map<std::string, double>& BenchmarkResults() {
  static std::map<std::string, double> cache;
  static bool ran = false;
  if (!ran) {
    struct Collect : benchmark::BenchmarkReporter {
      std::map<std::string, double>& out;
      explicit Collect(std::map<std::string, double>& m) : out(m) {}
      bool ReportContext(const Context&) override { return true; }
      void ReportRuns(const std::vector<Run>& runs) override {
        for (const auto& r : runs)
          out[r.benchmark_name()] = r.GetAdjustedCPUTime() / r.iterations * 1e9;
      }
    };
    Collect c(cache);
    benchmark::RunSpecifiedBenchmarks(&c);
    ran = true;
  }
  return cache;
}

inline double NsPerOp(const std::string& name) {
  const auto& r = BenchmarkResults();
  auto it = r.find(name);
  if (it == r.end()) {
    ADD_FAILURE() << "Benchmark not registered: " << name;
    return 0.0;
  }
  return it->second;
}

} // namespace PerfCompare

// A should be at least `factor` times faster than B.
#define EXPECT_AT_LEAST_N_TIMES_FASTER(A, B, factor)                           \
  do {                                                                         \
    double ta_ = ::PerfCompare::NsPerOp(#A);                                \
    double tb_ = ::PerfCompare::NsPerOp(#B);                                \
    EXPECT_GT(tb_ / ta_, (factor))                                             \
        << #A << " (" << ta_ << " ns) not " << (factor) << "x faster than "    \
        << #B << " (" << tb_ << " ns)";                                        \
  } while (0)

// A should not be slower than B (10% tolerance for measurement noise).
#define EXPECT_NOT_SLOWER_THAN(A, B) EXPECT_AT_LEAST_N_TIMES_FASTER(A, B, 0.9)

#endif // CPPINTEROP_TEST_PERFCOMPARE_H
