#ifndef ROOT7_RNTuple_Test_RXTuple
#define ROOT7_RNTuple_Test_RXTuple

#include <cstdint>
#include "Rtypes.h"

// NOTE: This namespace must be the same as RNTuple
namespace ROOT {

// A mock of the RNTuple class, used to write a "future version" of RNTuple to a file.
// The idea is:
//   1. we write a "RXTuple" to a file, with a schema identical to RNTuple + some
//      hypothetical future fields
//   2. we read back the file and patch the name to transmute the on-disk
//      binary to a serialized RNTuple with some additional fields
//   3. we read back the patched file to ensure the current version of RNTuple can handle
//      schema evolution in a fwd-compatible way.
// For ease of patching, the name of this struct has the same length as that of RNTuple.
class RXTuple final {
public:
   std::uint16_t fVersionEpoch = 9;
   std::uint16_t fVersionMajor = 9;
   std::uint16_t fVersionMinor = 9;
   std::uint16_t fVersionPatch = 9;
   // These values are arbitrary and here to provide easily recognizable patterns in the binary file.
   std::uint64_t fSeekHeader = 0x42;
   std::uint64_t fNBytesHeader = 0x99;
   std::uint64_t fLenHeader = 0x99;
   std::uint64_t fSeekFooter = 0x34;
   std::uint64_t fNBytesFooter = 0x66;
   std::uint64_t fLenFooter = 0x66;

   // Start of future fields
   std::uint16_t fFutureField0 = 0;
   std::uint32_t fFutureField1 = 1;
   std::uint64_t fFutureField2 = 2;
   std::uint64_t fFutureField3 = 3;

   // Use an unreasonably high class version so we're guaranteed to always be a future version
   // (but not too high! Putting 9999 would probably cause trouble due to its special meaning)
   ClassDefNV(RXTuple, 99);
}; // class RXTuple

} // namespace ROOT

#endif
