/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RProfile
#define ROOT_RProfile

#include "RAxisVariant.hxx"
#include "RBinIndex.hxx"
#include "RHistEngine.hxx"
#include "RRegularAxis.hxx"
#include "RWeight.hxx"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
A profile histogram, computing statistical quantities of an additional variable per bin.

Calling \ref Fill(const std::tuple<A...> &args, const V &value) "Fill" requires an additional value:
\code
ROOT::Experimental::RProfile profile(10, {5, 15});
profile.Fill(std::make_tuple(8.2), 23.0);
profile.Fill(std::make_tuple(8.7), 25.0);
// Bin 3 has a mean of 24.0 and a standard deviation of 1.0
\endcode

The class is not templated, the bin content is always of type RProfileBin.

An object can have arbitrary dimensionality determined at run-time. The axis configuration is passed as a vector of
RAxisVariant:
\code
std::vector<ROOT::Experimental::RAxisVariant> axes;
axes.push_back(ROOT::Experimental::RRegularAxis(10, {5, 15}));
axes.push_back(ROOT::Experimental::RVariableBinAxis({1, 10, 100, 1000}));
ROOT::Experimental::RProfile profile(axes);
// profile.GetNDimensions() will return 2
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RProfile final {
   struct RValueWrapper {
      double fValue;

      explicit RValueWrapper(double value) : fValue(value) {}
   };

   struct RValueWeightWrapper {
      double fValue;
      double fWeight;

      explicit RValueWeightWrapper(double value, double weight) : fValue(value), fWeight(weight) {}
   };

public:
   /// The bin content type of a profile histogram.
   struct RProfileBin final {
      double fSumValues = 0;
      double fSumValues2 = 0;
      double fSum = 0;
      double fSum2 = 0;

      RProfileBin &operator+=(const RValueWrapper &rhs)
      {
         fSumValues += rhs.fValue;
         fSumValues2 += rhs.fValue * rhs.fValue;
         fSum++;
         fSum2++;
         return *this;
      }

      RProfileBin &operator+=(const RValueWeightWrapper &rhs)
      {
         fSumValues += rhs.fWeight * rhs.fValue;
         fSumValues2 += rhs.fWeight * rhs.fValue * rhs.fValue;
         fSum += rhs.fWeight;
         fSum2 += rhs.fWeight * rhs.fWeight;
         return *this;
      }
   };

private:
   /// The histogram engine including the bin contents.
   RHistEngine<RProfileBin> fEngine;

public:
   /// Construct a profile histogram.
   ///
   /// \param[in] axes the axis objects, must have size > 0
   explicit RProfile(std::vector<RAxisVariant> axes) : fEngine(std::move(axes)) {}

   /// Construct a profile histogram.
   ///
   /// Note that there is no perfect forwarding of the axis objects. If that is needed, use the
   /// \ref RProfile(std::vector<RAxisVariant> axes) "overload accepting a std::vector".
   ///
   /// \param[in] axes the axis objects, must have size > 0
   explicit RProfile(std::initializer_list<RAxisVariant> axes) : RProfile(std::vector(axes)) {}

   /// Construct a profile histogram.
   ///
   /// Note that there is no perfect forwarding of the axis objects. If that is needed, use the
   /// \ref RProfile(std::vector<RAxisVariant> axes) "overload accepting a std::vector".
   ///
   /// \param[in] axis1 the first axis object
   /// \param[in] axes the remaining axis objects
   template <typename... Axes>
   explicit RProfile(const RAxisVariant &axis1, const Axes &...axes)
      : RProfile(std::vector<RAxisVariant>{axis1, axes...})
   {
   }

   /// Construct a one-dimensional profile histogram with a regular axis.
   ///
   /// \param[in] nNormalBins the number of normal bins, must be > 0
   /// \param[in] interval the axis interval (lower end inclusive, upper end exclusive)
   /// \par See also
   /// the \ref RRegularAxis::RRegularAxis(std::uint64_t nNormalBins, std::pair<double, double> interval, bool
   /// enableFlowBins) "constructor of RRegularAxis"
   RProfile(std::uint64_t nNormalBins, std::pair<double, double> interval)
      : RProfile(std::vector<RAxisVariant>{RRegularAxis(nNormalBins, interval)})
   {
   }

   /// The copy constructor is deleted.
   ///
   /// Copying all bin contents can be an expensive operation, depending on the number of bins. If required, users can
   /// explicitly call Clone().
   RProfile(const RProfile &) = delete;
   /// Efficiently move construct a profile histogram.
   ///
   /// After this operation, the moved-from object is invalid.
   RProfile(RProfile &&) = default;

   /// The copy assignment operator is deleted.
   ///
   /// Copying all bin contents can be an expensive operation, depending on the number of bins. If required, users can
   /// explicitly call Clone().
   RProfile &operator=(const RProfile &) = delete;
   /// Efficiently move a profile histogram.
   ///
   /// After this operation, the moved-from object is invalid.
   RProfile &operator=(RProfile &&) = default;

   ~RProfile() = default;

   /// \name Accessors
   /// \{

   const RHistEngine<RProfileBin> &GetEngine() const { return fEngine; }

   const std::vector<RAxisVariant> &GetAxes() const { return fEngine.GetAxes(); }
   std::size_t GetNDimensions() const { return fEngine.GetNDimensions(); }
   std::uint64_t GetTotalNBins() const { return fEngine.GetTotalNBins(); }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// std::array<ROOT::Experimental::RBinIndex, 2> indices = {3, 5};
   /// const auto &content = profile.GetBinContent(indices);
   /// \endcode
   ///
   /// \note Compared to TH1 conventions, the first normal bin has index 0 and underflow and overflow bins are special
   /// values. See also the class documentation of RBinIndex.
   ///
   /// Throws an exception if the number of indices does not match the axis configuration or the bin is not found.
   ///
   /// \param[in] indices the array of indices for each axis
   /// \return the bin content
   /// \par See also
   /// the \ref GetBinContent(const A &... args) const "variadic function template overload" accepting arguments
   /// directly
   template <std::size_t N>
   const RProfileBin &GetBinContent(const std::array<RBinIndex, N> &indices) const
   {
      return fEngine.GetBinContent(indices);
   }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// std::vector<ROOT::Experimental::RBinIndex> indices = {3, 5};
   /// const auto &content = profile.GetBinContent(indices);
   /// \endcode
   ///
   /// \note Compared to TH1 conventions, the first normal bin has index 0 and underflow and overflow bins are special
   /// values. See also the class documentation of RBinIndex.
   ///
   /// Throws an exception if the number of indices does not match the axis configuration or the bin is not found.
   ///
   /// \param[in] indices the array of indices for each axis
   /// \return the bin content
   /// \par See also
   /// the \ref GetBinContent(const A &... args) const "variadic function template overload" accepting arguments
   /// directly
   const RProfileBin &GetBinContent(const std::vector<RBinIndex> &indices) const
   {
      return fEngine.GetBinContent(indices);
   }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// const auto &content = profile.GetBinContent(3, 5);
   /// \endcode
   ///
   /// \note Compared to TH1 conventions, the first normal bin has index 0 and underflow and overflow bins are special
   /// values. See also the class documentation of RBinIndex.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration or the bin is not found.
   ///
   /// \param[in] args the arguments for each axis
   /// \return the bin content
   /// \par See also
   /// the function overloads accepting \ref GetBinContent(const std::array<RBinIndex, N> &indices) const "`std::array`"
   /// or \ref GetBinContent(const std::vector<RBinIndex> &indices) const "`std::vector`"
   template <typename... A>
   const RProfileBin &GetBinContent(const A &...args) const
   {
      return fEngine.GetBinContent(args...);
   }

   /// \}
   /// \name Filling
   /// \{

   /// Fill an entry into the profile histogram.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// auto args = std::make_tuple(8.5, 10.5);
   /// profile.Fill(args, 23.0);
   /// \endcode
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \param[in] v the additional argument
   /// \par See also
   /// the \ref Fill(const std::tuple<A...> &args, const V &value, RWeight weight) "overload for weighted filling"
   template <typename... A, typename V>
   void Fill(const std::tuple<A...> &args, const V &value)
   {
      RValueWrapper wrapper(value);
      fEngine.Fill(args, wrapper);
   }

   /// Fill an entry into the profile histogram with a weight.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// auto args = std::make_tuple(8.5, 10.5);
   /// profile.Fill(args, 23.0, ROOT::Experimental::RWeight(0.8));
   /// \endcode
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \param[in] v the additional argument
   /// \param[in] weight the weight for this entry
   /// \par See also
   /// the \ref Fill(const std::tuple<A...> &args, const V &value) "overload for unweighted filling"
   template <typename... A, typename V>
   void Fill(const std::tuple<A...> &args, const V &value, RWeight weight)
   {
      RValueWeightWrapper wrapper(value, weight.fValue);
      fEngine.Fill(args, wrapper);
   }

   /// \}

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RProfile"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
