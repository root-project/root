/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RAxes
#define ROOT_RAxes

#include "RBinIndex.hxx"
#include "RLinearizedIndex.hxx"
#include "RRegularAxis.hxx"
#include "RVariableBinAxis.hxx"

#include <array>
#include <cassert>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/// Variant of all supported axis types.
using RAxisVariant = std::variant<RRegularAxis, RVariableBinAxis>;

namespace Internal {

/**
Bin configurations for all dimensions of a histogram.
*/
class RAxes final {
   std::vector<RAxisVariant> fAxes;

public:
   /// \param[in] axes the axis objects, must have size > 0
   explicit RAxes(std::vector<RAxisVariant> axes) : fAxes(std::move(axes))
   {
      if (fAxes.empty()) {
         throw std::invalid_argument("must have at least 1 axis object");
      }
   }

   std::size_t GetNDimensions() const { return fAxes.size(); }
   const std::vector<RAxisVariant> &Get() const { return fAxes; }

   friend bool operator==(const RAxes &lhs, const RAxes &rhs) { return lhs.fAxes == rhs.fAxes; }

   /// Compute the total number of bins for all axes.
   ///
   /// It is the product of each dimension's total number of bins.
   ///
   /// \return the total number of bins
   std::size_t ComputeTotalNBins() const
   {
      std::size_t totalNBins = 1;
      for (auto &&axis : fAxes) {
         if (auto *regular = std::get_if<RRegularAxis>(&axis)) {
            totalNBins *= regular->GetTotalNBins();
         } else if (auto *variable = std::get_if<RVariableBinAxis>(&axis)) {
            totalNBins *= variable->GetTotalNBins();
         } else {
            throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
         }
      }
      return totalNBins;
   }

private:
   template <std::size_t I, typename... A>
   RLinearizedIndex ComputeGlobalIndex(std::size_t index, const std::tuple<A...> &args) const
   {
      const auto &axis = fAxes[I];
      RLinearizedIndex linIndex;
      if (auto *regular = std::get_if<RRegularAxis>(&axis)) {
         index *= regular->GetTotalNBins();
         linIndex = regular->ComputeLinearizedIndex(std::get<I>(args));
      } else if (auto *variable = std::get_if<RVariableBinAxis>(&axis)) {
         index *= variable->GetTotalNBins();
         linIndex = variable->ComputeLinearizedIndex(std::get<I>(args));
      } else {
         throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
      }
      if (!linIndex.fValid) {
         return {0, false};
      }
      index += linIndex.fIndex;
      if constexpr (I + 1 < sizeof...(A)) {
         return ComputeGlobalIndex<I + 1>(index, args);
      }
      return {index, true};
   }

public:
   /// Compute the global index for all axes.
   ///
   /// \param[in] args the arguments
   /// \return the global index that may be invalid
   template <typename... A>
   RLinearizedIndex ComputeGlobalIndex(const std::tuple<A...> &args) const
   {
      if (sizeof...(A) != fAxes.size()) {
         throw std::invalid_argument("invalid number of arguments to ComputeGlobalIndex");
      }
      return ComputeGlobalIndex<0, A...>(0, args);
   }

   /// Compute the global index for all axes.
   ///
   /// \param[in] indices the array of RBinIndex
   /// \return the global index that may be invalid
   template <std::size_t N>
   RLinearizedIndex ComputeGlobalIndex(const std::array<RBinIndex, N> &indices) const
   {
      if (N != fAxes.size()) {
         throw std::invalid_argument("invalid number of indices passed to ComputeGlobalIndex");
      }
      std::size_t globalIndex = 0;
      for (std::size_t i = 0; i < N; i++) {
         const auto &index = indices[i];
         const auto &axis = fAxes[i];
         RLinearizedIndex linIndex;
         if (auto *regular = std::get_if<RRegularAxis>(&axis)) {
            globalIndex *= regular->GetTotalNBins();
            linIndex = regular->GetLinearizedIndex(index);
         } else if (auto *variable = std::get_if<RVariableBinAxis>(&axis)) {
            globalIndex *= variable->GetTotalNBins();
            linIndex = variable->GetLinearizedIndex(index);
         } else {
            throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
         }
         if (!linIndex.fValid) {
            return {0, false};
         }
         globalIndex += linIndex.fIndex;
      }
      return {globalIndex, true};
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RAxes"); }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
