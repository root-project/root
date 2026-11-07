/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RSliceSpec
#define ROOT_RSliceSpec

#include "RBinIndexRange.hxx"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ROOT {
namespace Experimental {

/**
Specification of a slice operation along one dimension.

\code
using ROOT::Experimental::RSliceSpec;
// When not specifying a range, the slice will include all bins.
RSliceSpec full;
// In the following, assuming range is an RBinIndexRange.
RSliceSpec slice(range);

// Operations are specified with parameters.
RSliceSpec rebin(RSliceSpec::ROperationRebin(2));
RSliceSpec sum(RSliceSpec::ROperationSum{});

// Finally, it is possible to combine a range and an operation.
RSliceSpec sliceRebin(range, RSliceSpec::ROperationRebin(2));
RSliceSpec sliceSum(range, RSliceSpec::ROperationSum{});
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RSliceSpec final {
public:
   /// Rebin the dimension, grouping a number of original bins into a new one.
   class ROperationRebin final {
      std::uint64_t fNGroup = 1;

   public:
      /// \param[in] nGroup the number of bins to group, must be > 0
      ROperationRebin(std::uint64_t nGroup) : fNGroup(nGroup)
      {
         if (nGroup == 0) {
            throw std::invalid_argument("nGroup must be > 0");
         }
      }

      std::uint64_t GetNGroup() const { return fNGroup; }
   };

   /// Sum bins along this dimension, effectively resulting in a projection.
   class ROperationSum final {
      // empty, no parameters at the moment
   };

private:
   /// The range of the slice; can be invalid to signify the full range
   RBinIndexRange fRange;
   /// The operation to perform, if any
   std::variant<std::monostate, ROperationRebin, ROperationSum> fOperation;

public:
   /// A default slice operation that keeps the dimension untouched.
   RSliceSpec() = default;

   /// A slice of a dimension.
   ///
   /// \param[in] range the range of the slice
   RSliceSpec(RBinIndexRange range) : fRange(std::move(range)) {}

   /// A rebin operation of a dimension.
   RSliceSpec(ROperationRebin rebin) : fOperation(std::move(rebin)) {}

   /// A sum operation of a dimension.
   RSliceSpec(ROperationSum sum) : fOperation(std::move(sum)) {}

   /// A rebin operation of a slice of the dimension.
   RSliceSpec(RBinIndexRange range, ROperationRebin rebin) : fRange(std::move(range)), fOperation(std::move(rebin)) {}

   /// A sum operation of a slice of the dimension.
   RSliceSpec(RBinIndexRange range, ROperationSum sum) : fRange(std::move(range)), fOperation(std::move(sum)) {}

   const RBinIndexRange &GetRange() const { return fRange; }
   bool HasOperation() const { return !std::holds_alternative<std::monostate>(fOperation); }
   const ROperationRebin *GetOperationRebin() const { return std::get_if<ROperationRebin>(&fOperation); }
   const ROperationSum *GetOperationSum() const { return std::get_if<ROperationSum>(&fOperation); }
};

} // namespace Experimental
} // namespace ROOT

#endif
