// Test for ROOT-8894
// We check if

#include "ROOT/RDataFrame.hxx"
#include <limits>

int main() {
#if defined(NDEBUG) || defined(__aarch64__) || !defined(R__B64)
 return 0;
#else
  ULong64_t nEntries = std::numeric_limits<unsigned int>::max();
  nEntries += 128;

  ROOT::RDataFrame f(nEntries);

  double iEntry = 0.;
  auto d = f.Define("iEntry", [&iEntry]() { return iEntry++; });
  auto histo =
      d.Histo1D<double>({"", "", 128, 0, (double)nEntries}, {"iEntry"});

  auto countedEntries = *f.Count();
  return nEntries == countedEntries ? 0 : 1;
#endif
}
