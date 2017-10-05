// Test for ROOT-8894
// We check if

#include "ROOT/TDataFrame.hxx"
#include <limits>

int main() {
#if !defined(NDEBUG) && !defined(__aarch64__) && defined(R__B64)
  ULong64_t nEntries = std::numeric_limits<unsigned int>::max();
  nEntries += 128;

  ROOT::Experimental::TDataFrame f(nEntries);

  double iEntry = 0.;
  auto d = f.Define("iEntry", [&iEntry]() { return iEntry++; });
  auto histo =
      d.Histo1D<double>({"", "", 128, 0, (double)nEntries}, {"iEntry"});

  auto countedEntries = *f.Count();
  return nEntries == countedEntries ? 0 : 1;
#else
  return 0;
#endif
}
