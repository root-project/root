/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RAxisVariant
#define ROOT_RAxisVariant

#include "RCategoricalAxis.hxx"
#include "RRegularAxis.hxx"
#include "RVariableBinAxis.hxx"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
A variant of all supported axis types.

This class provides easy access to the contained axis object and dispatching methods for common accessors.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RAxisVariant final {
public:
   using VariantType = std::variant<RRegularAxis, RVariableBinAxis, RCategoricalAxis>;

private:
   VariantType fVariant;

public:
   RAxisVariant(VariantType axis) : fVariant(std::move(axis)) {}
   RAxisVariant(RRegularAxis axis) : fVariant(std::move(axis)) {}
   RAxisVariant(RVariableBinAxis axis) : fVariant(std::move(axis)) {}
   RAxisVariant(RCategoricalAxis axis) : fVariant(std::move(axis)) {}

   const VariantType &GetVariant() const { return fVariant; }

   /// \return the RRegularAxis or nullptr, if this variant stores a different axis type
   const RRegularAxis *GetRegularAxis() const { return std::get_if<RRegularAxis>(&fVariant); }
   /// \return the RVariableBinAxis or nullptr, if this variant stores a different axis type
   const RVariableBinAxis *GetVariableBinAxis() const { return std::get_if<RVariableBinAxis>(&fVariant); }
   /// \return the RCategoricalAxis or nullptr, if this variant stores a different axis type
   const RCategoricalAxis *GetCategoricalAxis() const { return std::get_if<RCategoricalAxis>(&fVariant); }

   std::uint64_t GetNNormalBins() const
   {
      if (auto *regular = GetRegularAxis()) {
         return regular->GetNNormalBins();
      } else if (auto *variable = GetVariableBinAxis()) {
         return variable->GetNNormalBins();
      } else if (auto *categorical = GetCategoricalAxis()) {
         return categorical->GetNNormalBins();
      } else {
         throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
      }
   }

   std::uint64_t GetTotalNBins() const
   {
      if (auto *regular = GetRegularAxis()) {
         return regular->GetTotalNBins();
      } else if (auto *variable = GetVariableBinAxis()) {
         return variable->GetTotalNBins();
      } else if (auto *categorical = GetCategoricalAxis()) {
         return categorical->GetTotalNBins();
      } else {
         throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
      }
   }

   /// Get the range of all normal bins.
   ///
   /// \return the bin index range from the first to the last normal bin, inclusive
   RBinIndexRange GetNormalRange() const
   {
      if (auto *regular = GetRegularAxis()) {
         return regular->GetNormalRange();
      } else if (auto *variable = GetVariableBinAxis()) {
         return variable->GetNormalRange();
      } else if (auto *categorical = GetCategoricalAxis()) {
         return categorical->GetNormalRange();
      } else {
         throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
      }
   }

   /// Get a range of normal bins.
   ///
   /// \param[in] begin the begin of the bin index range (inclusive), must be normal
   /// \param[in] end the end of the bin index range (exclusive), must be normal and >= begin
   /// \return a bin index range \f$[begin, end)\f$
   RBinIndexRange GetNormalRange(RBinIndex begin, RBinIndex end) const
   {
      if (auto *regular = GetRegularAxis()) {
         return regular->GetNormalRange(begin, end);
      } else if (auto *variable = GetVariableBinAxis()) {
         return variable->GetNormalRange(begin, end);
      } else if (auto *categorical = GetCategoricalAxis()) {
         return categorical->GetNormalRange(begin, end);
      } else {
         throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
      }
   }

   /// Get the full range of all bins.
   ///
   /// This includes underflow and overflow bins, if enabled.
   ///
   /// \return the bin index range of all bins
   RBinIndexRange GetFullRange() const
   {
      if (auto *regular = GetRegularAxis()) {
         return regular->GetFullRange();
      } else if (auto *variable = GetVariableBinAxis()) {
         return variable->GetFullRange();
      } else if (auto *categorical = GetCategoricalAxis()) {
         return categorical->GetFullRange();
      } else {
         throw std::logic_error("unimplemented axis type"); // GCOVR_EXCL_LINE
      }
   }

   friend bool operator==(const RAxisVariant &lhs, const RAxisVariant &rhs) { return lhs.fVariant == rhs.fVariant; }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RAxisVariant"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
