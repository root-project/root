// The problem appears when generating dictionaries for variadic templates with 2 vs. 3 arguments
// Mostly copied and adapted from CMSSW
template <int A = 64, bool B = false>
struct TestSoALayout {};

template <typename T0, typename... Args>
struct PortableHostMultiCollection {};

using TestSoA = TestSoALayout<>;

namespace portabletest {
using TestHostMultiCollection2 = PortableHostMultiCollection<TestSoA, TestSoA>;
using TestHostMultiCollection3 = PortableHostMultiCollection<TestSoA, TestSoA, TestSoA>;
} // namespace portabletest
