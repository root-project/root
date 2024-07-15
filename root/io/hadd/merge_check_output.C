#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <Compression.h>

#include <TSystem.h>

#include <string>
#include <utility>
#include <sstream>
#include <cstdint>
#include <cstring>

// Copy-pasted from gtest-internal.h
namespace {
const uint32_t kMaxUlps = 4;
const size_t kBitCount = 8 * sizeof(float);
const uint32_t kSignBitMask = static_cast<uint32_t>(1) << (kBitCount - 1);

uint32_t SignAndMagnitudeToBiased(const uint32_t &sam)
{
   if (kSignBitMask & sam) {
      // sam represents a negative number.
      return ~sam + 1;
   } else {
      // sam represents a positive number.
      return kSignBitMask | sam;
   }
}

uint32_t DistanceBetweenSignAndMagnitudeNumbers(const uint32_t &sam1, const uint32_t &sam2)
{
   const uint32_t biased1 = SignAndMagnitudeToBiased(sam1);
   const uint32_t biased2 = SignAndMagnitudeToBiased(sam2);
   return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

bool FloatAlmostEquals(float a, float b)
{
   uint32_t ia, ib;
   memcpy(&ia, &a, sizeof(float));
   memcpy(&ib, &b, sizeof(float));
   return DistanceBetweenSignAndMagnitudeNumbers(ia, ib) <= kMaxUlps;
}
} // namespace

int merge_check_output(int expectedCompression, const char *fnameOut, const char *fnameIn1,
                                  const char *fnameIn2)
{
   using namespace ROOT::Experimental;

   auto noPrereleaseWarning = RLogScopedVerbosity(NTupleLog(), ROOT::Experimental::ELogLevel::kError);

   std::stringstream errs;
   {
      auto ntuple = RNTupleReader::Open("ntpl", fnameOut);
      auto viewI = ntuple->GetView<int>("I");
      auto viewF = ntuple->GetView<float>("F");
      if (auto v = viewI(0); v != 1337)
         errs << "Expected v to be 1337 but it's " << v << "\n";
      if (auto v = viewI(1); v != 123)
         errs << "Expected v to be 123 but it's " << v << "\n";
      if (auto v = viewF(0); !FloatAlmostEquals(v, 666.f))
         errs << "Expected v to be 666.f but it's " << v << "\n";
      if (auto v = viewF(1); !FloatAlmostEquals(v, 420.f))
         errs << "Expected v to be 420.f but it's " << v << "\n";
   }

   gSystem->Unlink(fnameOut);
   gSystem->Unlink(fnameIn1);
   gSystem->Unlink(fnameIn2);

   auto errsStr = errs.str();
   if (errsStr.length() > 0) {
      std::cerr << errsStr << "\n";
      return 1;
   }

   return 0;
}
