/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RCategoricalAxis
#define ROOT_RCategoricalAxis

#include "RBinIndex.hxx"
#include "RBinIndexRange.hxx"
#include "RLinearizedIndex.hxx"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
An axis with categorical bins.

For example, the following creates an axis with 3 categories:
\code
std::vector<std::string> categories = {"a", "b", "c"};
ROOT::Experimental::RCategoricalAxis axis(categories);
\endcode

It is possible to disable the overflow bin by passing `enableOverflowBin = false`. In that case, arguments outside the
axis will be silently discarded.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RCategoricalAxis final {
public:
   using ArgumentType = std::string_view;

private:
   /// The categories as defined by the user
   std::vector<std::string> fCategories;
   /// Whether the overflow bin is enabled
   bool fEnableOverflowBin;

public:
   /// Construct an axis object with categories.
   ///
   /// \param[in] categories the categories without duplicates, must define at least one bin (i.e. size >= 1)
   /// \param[in] enableOverflowBin whether to enable the overflow bin
   explicit RCategoricalAxis(std::vector<std::string> categories, bool enableOverflowBin = true)
      : fCategories(std::move(categories)), fEnableOverflowBin(enableOverflowBin)
   {
      if (fCategories.size() < 1) {
         throw std::invalid_argument("must have at least one category");
      }
      // Check for duplicates, use std::string_view to avoid copying the category strings.
      std::unordered_set<std::string_view> set;
      for (std::size_t i = 0; i < fCategories.size(); i++) {
         if (!set.insert(fCategories[i]).second) {
            std::string msg = "duplicate category '" + fCategories[i] + "' for bin " + std::to_string(i);
            throw std::invalid_argument(msg);
         }
      }
   }

   std::size_t GetNNormalBins() const { return fCategories.size(); }
   std::size_t GetTotalNBins() const { return fEnableOverflowBin ? fCategories.size() + 1 : fCategories.size(); }
   const std::vector<std::string> &GetCategories() const { return fCategories; }
   bool HasOverflowBin() const { return fEnableOverflowBin; }

   friend bool operator==(const RCategoricalAxis &lhs, const RCategoricalAxis &rhs)
   {
      return lhs.fCategories == rhs.fCategories && lhs.fEnableOverflowBin == rhs.fEnableOverflowBin;
   }

   /// Compute the linarized index for a single argument.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fCategories.size() - 1\f$ and the overflow bin has index
   /// \f$fCategories.size()\f$. If the argument is not a recognized category and the overflow bin is disabled, the
   /// return value is invalid.
   ///
   /// \param[in] x the argument
   /// \return the linearized index that may be invalid
   RLinearizedIndex ComputeLinearizedIndex(std::string_view x) const
   {
      // FIXME: Optimize with hashing... (?)
      for (std::size_t bin = 0; bin < fCategories.size(); bin++) {
         if (fCategories[bin] == x) {
            return {bin, true};
         }
      }

      // Category not found
      return {fCategories.size(), fEnableOverflowBin};
   }

   /// Get the linearized index for an RBinIndex.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fCategories.size() - 1\f$ and the overflow bin has index
   /// \f$fCategories.size()\f$.
   ///
   /// \param[in] index the RBinIndex
   /// \return the linearized index that may be invalid
   RLinearizedIndex GetLinearizedIndex(RBinIndex index) const
   {
      if (index.IsUnderflow()) {
         // No underflow bin for RCategoricalAxis...
         return {0, false};
      } else if (index.IsOverflow()) {
         return {fCategories.size(), fEnableOverflowBin};
      } else if (index.IsInvalid()) {
         return {0, false};
      }
      assert(index.IsNormal());
      std::size_t bin = index.GetIndex();
      return {bin, bin < fCategories.size()};
   }

   /// Get the range of all normal bins.
   ///
   /// \return the bin index range from the first to the last normal bin, inclusive
   RBinIndexRange GetNormalRange() const
   {
      return Internal::CreateBinIndexRange(RBinIndex(0), RBinIndex(fCategories.size()), 0);
   }

   /// Get a range of normal bins.
   ///
   /// \param[in] begin the begin of the bin index range (inclusive), must be normal
   /// \param[in] end the end of the bin index range (exclusive), must be normal and >= begin
   /// \return a bin index range \f$[begin, end)\f$
   RBinIndexRange GetNormalRange(RBinIndex begin, RBinIndex end) const
   {
      if (!begin.IsNormal()) {
         throw std::invalid_argument("begin must be a normal bin");
      }
      if (begin.GetIndex() >= fCategories.size()) {
         throw std::invalid_argument("begin must be inside the axis");
      }
      if (!end.IsNormal()) {
         throw std::invalid_argument("end must be a normal bin");
      }
      if (end.GetIndex() > fCategories.size()) {
         throw std::invalid_argument("end must be inside or past the axis");
      }
      if (!(end >= begin)) {
         throw std::invalid_argument("end must be >= begin");
      }
      return Internal::CreateBinIndexRange(begin, end, 0);
   }

   /// Get the full range of all bins.
   ///
   /// This includes the overflow bin, if enabled.
   ///
   /// \return the bin index range of all bins
   RBinIndexRange GetFullRange() const
   {
      return fEnableOverflowBin ? Internal::CreateBinIndexRange(RBinIndex(0), RBinIndex(), fCategories.size())
                                : GetNormalRange();
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RCategoricalAxis"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
