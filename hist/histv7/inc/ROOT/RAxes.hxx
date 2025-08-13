/// \file
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT_RAxes
#define ROOT_RAxes

#include "RLinearizedIndex.hxx"
#include "RRegularAxis.hxx"
#include "RVariableBinAxis.hxx"

#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {
namespace Internal {

/**
Bin configurations for all dimensions of a histogram.
*/
class RAxes final {
public:
   using AxisVariant = std::variant<RRegularAxis, RVariableBinAxis>;

private:
   std::vector<AxisVariant> fAxes;

public:
   /// \param[in] axes the axis objects, must have size > 0
   explicit RAxes(std::vector<AxisVariant> axes) : fAxes(std::move(axes))
   {
      if (fAxes.empty()) {
         throw std::invalid_argument("must have at least 1 axis object");
      }
   }

   std::size_t GetNumDimensions() const { return fAxes.size(); }
   const std::vector<AxisVariant> &Get() const { return fAxes; }

   friend bool operator==(const RAxes &lhs, const RAxes &rhs) { return lhs.fAxes == rhs.fAxes; }

   /// Compute the total number of bins for all axes.
   ///
   /// It is the product of each dimension's total number of bins.
   ///
   /// \return the total number of bins
   std::size_t ComputeTotalNumBins() const
   {
      std::size_t totalNumBins = 1;
      for (auto &&axis : fAxes) {
         if (auto *regular = std::get_if<RRegularAxis>(&axis)) {
            totalNumBins *= regular->GetTotalNumBins();
         } else if (auto *variable = std::get_if<RVariableBinAxis>(&axis)) {
            totalNumBins *= variable->GetTotalNumBins();
         } else {
            throw std::logic_error("unimplemented axis type");
         }
      }
      return totalNumBins;
   }

private:
   template <std::size_t I, typename... A>
   RLinearizedIndex ComputeGlobalIndex(std::size_t index, const std::tuple<A...> &args) const
   {
      const auto &axis = fAxes[I];
      RLinearizedIndex linIndex;
      if (auto *regular = std::get_if<RRegularAxis>(&axis)) {
         index *= regular->GetTotalNumBins();
         linIndex = regular->ComputeLinearizedIndex(std::get<I>(args));
      } else if (auto *variable = std::get_if<RVariableBinAxis>(&axis)) {
         index *= variable->GetTotalNumBins();
         linIndex = variable->ComputeLinearizedIndex(std::get<I>(args));
      } else {
         throw std::logic_error("unimplemented axis type");
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

   /// ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RAxes"); }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
