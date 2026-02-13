/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RSliceBinIndexMapper
#define ROOT_RSliceBinIndexMapper

#include "RBinIndex.hxx"
#include "RBinIndexRange.hxx"
#include "RSliceSpec.hxx"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {

/**
Mapper of bin indices for slice operations.
*/
class RSliceBinIndexMapper final {
   /// The requested slice specifications
   std::vector<RSliceSpec> fSliceSpecs;
   /// The expected dimensionality of the mapped indices
   std::size_t fMappedDimensionality;

   static std::size_t ComputeMappedDimensionality(const std::vector<RSliceSpec> &sliceSpecs)
   {
      std::size_t dimensionality = 0;
      for (auto &&spec : sliceSpecs) {
         // A sum operation makes the dimension disappear.
         if (spec.GetOperationSum() == nullptr) {
            dimensionality++;
         }
      }
      return dimensionality;
   }

public:
   /// \param[in] sliceSpecs the slice specifications, must have size > 0
   explicit RSliceBinIndexMapper(std::vector<RSliceSpec> sliceSpecs)
      : fSliceSpecs(std::move(sliceSpecs)), fMappedDimensionality(ComputeMappedDimensionality(fSliceSpecs))
   {
      if (fSliceSpecs.empty()) {
         throw std::invalid_argument("must have at least 1 slice specification");
      }
   }

   const std::vector<RSliceSpec> &GetSliceSpecs() const { return fSliceSpecs; }
   std::size_t GetMappedDimensionality() const { return fMappedDimensionality; }

   /// Map a vector of RBinIndex according to the slice specifications.
   ///
   /// \param[in] original the original bin indices
   /// \param[out] mapped the mapped bin indices
   /// \return whether the mapping was successful or the bin content should be discarded
   bool Map(const std::vector<RBinIndex> &original, std::vector<RBinIndex> &mapped) const
   {
      if (original.size() != fSliceSpecs.size()) {
         throw std::invalid_argument("invalid number of original indices passed to RSliceBinIndexMapper::Map");
      }
      if (mapped.size() != fMappedDimensionality) {
         throw std::invalid_argument("invalid size of mapped indices passed to RSliceBinIndexMapper::Map");
      }

      std::size_t mappedPos = 0;
      for (std::size_t i = 0; i < original.size(); i++) {
         RBinIndex index = original[i];
         if (index.IsInvalid()) {
            throw std::invalid_argument("invalid bin index passed to RSliceBinIndexMapper::Map");
         }

         const RSliceSpec &sliceSpec = fSliceSpecs[i];
         const auto &range = sliceSpec.GetRange();
         if (!range.IsInvalid()) {
            // For the moment, we only need to look at normal indices. Underflow and overflow indices map to themselves.
            if (index.IsNormal()) {
               const auto &begin = range.GetBegin();
               const auto &end = range.GetEnd();
               if (begin.IsNormal() && index < begin) {
                  index = RBinIndex::Underflow();
               } else if (end.IsNormal() && index >= end) {
                  index = RBinIndex::Overflow();
               } else if (begin.IsNormal()) {
                  // This normal bin is contained in the range. Its index must be shifted according to the begin of the
                  // range.
                  index -= begin.GetIndex();
                  assert(!index.IsInvalid());
               }
            }
         }

         if (auto *opRebin = sliceSpec.GetOperationRebin()) {
            if (index.IsNormal()) {
               index = RBinIndex(index.GetIndex() / opRebin->GetNGroup());
            }
         }

         mapped[mappedPos] = index;
         mappedPos++;
      }

      // If we got here, the loop should have filled all mapped indices.
      assert(mappedPos == mapped.size());
      return true;
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
