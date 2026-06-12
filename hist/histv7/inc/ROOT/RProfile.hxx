/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RProfile
#define ROOT_RProfile

#include "RAxisVariant.hxx"
#include "RBinIndex.hxx"
#include "RBinIndexMultiDimRange.hxx"
#include "RHistEngine.hxx"
#include "RHistStats.hxx"
#include "RHistUtils.hxx"
#include "RRegularAxis.hxx"
#include "RWeight.hxx"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
A profile histogram, computing statistical quantities of an additional variable per bin.

Calling \ref Fill(const A &... args) "Fill" requires an additional value:
\code
ROOT::Experimental::RProfile profile(10, {5, 15});
hist.Fill(8.2, 23.0);
hist.Fill(8.7, 25.0);
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

      RProfileBin &operator+=(const RProfileBin &rhs)
      {
         fSumValues += rhs.fSumValues;
         fSumValues2 += rhs.fSumValues2;
         fSum += rhs.fSum;
         fSum2 += rhs.fSum2;
         return *this;
      }

      RProfileBin &operator*=(double factor)
      {
         fSumValues *= factor;
         fSumValues2 *= factor;
         fSum *= factor;
         fSum2 *= factor * factor;
         return *this;
      }

      /// Add another bin content using atomic instructions.
      ///
      /// \param[in] rhs another bin content that must not be modified during the operation
      void AtomicAdd(const RProfileBin &rhs)
      {
         Internal::AtomicAdd(&fSumValues, rhs.fSumValues);
         Internal::AtomicAdd(&fSumValues2, rhs.fSumValues2);
         Internal::AtomicAdd(&fSum, rhs.fSum);
         Internal::AtomicAdd(&fSum2, rhs.fSum2);
      }
   };

private:
   /// The histogram engine including the bin contents.
   RHistEngine<RProfileBin> fEngine;
   /// The global histogram statistics.
   RHistStats fStats;

   /// Private constructor based off an engine.
   RProfile(RHistEngine<RProfileBin> engine) : fEngine(std::move(engine)), fStats(fEngine.GetNDimensions() + 1) {}

public:
   /// Construct a profile histogram.
   ///
   /// \param[in] axes the axis objects, must have size > 0
   explicit RProfile(std::vector<RAxisVariant> axes) : fEngine(std::move(axes)), fStats(fEngine.GetNDimensions() + 1)
   {
      // The axes parameter was moved, use from the engine.
      const auto &engineAxes = fEngine.GetAxes();
      for (std::size_t i = 0; i < engineAxes.size(); i++) {
         if (engineAxes[i].GetCategoricalAxis() != nullptr) {
            fStats.DisableDimension(i);
         }
      }
   }

   /// Construct a profile histogram.
   ///
   /// Note that there is no perfect forwarding of the axis objects. If that is needed, use the
   /// \ref RHist(std::vector<RAxisVariant> axes) "overload accepting a std::vector".
   ///
   /// \param[in] axes the axis objects, must have size > 0
   explicit RProfile(std::initializer_list<RAxisVariant> axes) : RProfile(std::vector(axes)) {}

   /// Construct a profile histogram.
   ///
   /// Note that there is no perfect forwarding of the axis objects. If that is needed, use the
   /// \ref RHist(std::vector<RAxisVariant> axes) "overload accepting a std::vector".
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
   const RHistStats &GetStats() const { return fStats; }

   const std::vector<RAxisVariant> &GetAxes() const { return fEngine.GetAxes(); }
   std::size_t GetNDimensions() const { return fEngine.GetNDimensions(); }
   std::uint64_t GetTotalNBins() const { return fEngine.GetTotalNBins(); }

   std::uint64_t GetNEntries() const { return fStats.GetNEntries(); }

   /// \}
   /// \name Computations
   /// \{

   /// \copydoc RHistStats::ComputeNEffectiveEntries()
   double ComputeNEffectiveEntries() const { return fStats.ComputeNEffectiveEntries(); }
   /// \copydoc RHistStats::ComputeMean()
   double ComputeMean(std::size_t dim = 0) const { return fStats.ComputeMean(dim); }
   /// \copydoc RHistStats::ComputeStdDev()
   double ComputeStdDev(std::size_t dim = 0) const { return fStats.ComputeStdDev(dim); }

   /// \}
   /// \name Accessors
   /// \{

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

   /// Get the multidimensional range of all bins.
   ///
   /// \return the multidimensional range
   RBinIndexMultiDimRange GetFullMultiDimRange() const { return fEngine.GetFullMultiDimRange(); }

   /// Set the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// std::array<ROOT::Experimental::RBinIndex, 2> indices = {3, 5};
   /// ROOT::Experimental::RProfile::RProfileBin value = /* ... */;
   /// profile.SetBinContent(indices, value);
   /// \endcode
   ///
   /// \note Compared to TH1 conventions, the first normal bin has index 0 and underflow and overflow bins are special
   /// values. See also the class documentation of RBinIndex.
   ///
   /// Throws an exception if the number of indices does not match the axis configuration or the bin is not found.
   ///
   /// \warning Setting the bin content will taint the global histogram statistics. Attempting to access its values, for
   /// example calling GetNEntries(), will throw exceptions.
   ///
   /// \param[in] indices the array of indices for each axis
   /// \param[in] value the new value of the bin content
   /// \par See also
   /// the \ref SetBinContent(const A &... args) "variadic function template overload" accepting arguments directly
   template <std::size_t N, typename V>
   void SetBinContent(const std::array<RBinIndex, N> &indices, const V &value)
   {
      fEngine.SetBinContent(indices, value);
      fStats.Taint();
   }

   /// Set the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// ROOT::Experimental::RProfile::RProfileBin value = /* ... */;
   /// profile.SetBinContent(3, 5, value);
   /// \endcode
   ///
   /// \note Compared to TH1 conventions, the first normal bin has index 0 and underflow and overflow bins are special
   /// values. See also the class documentation of RBinIndex.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration or the bin is not found.
   ///
   /// \warning Setting the bin content will taint the global histogram statistics. Attempting to access its values, for
   /// example calling GetNEntries(), will throw exceptions.
   ///
   /// \param[in] args the arguments for each axis and the new value of the bin content
   /// \par See also
   /// the \ref SetBinContent(const std::array<RBinIndex, N> &indices, const V &value) "function overload" accepting
   /// `std::array`
   template <typename... A>
   void SetBinContent(const A &...args)
   {
      fEngine.SetBinContent(args...);
      fStats.Taint();
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
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly and the
   /// \ref Fill(const std::tuple<A...> &args, const V &value, RWeight weight) "overload for weighted filling"
   template <typename... A, typename V>
   void Fill(const std::tuple<A...> &args, const V &value)
   {
      RValueWrapper wrapper(value);
      fEngine.Fill(args, wrapper);
      fStats.Fill(std::tuple_cat(args, std::make_tuple(wrapper.fValue)));
   }

   /// Fill an entry into the profile histogram with a weight.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// auto args = std::make_tuple(8.5, 10.5);
   /// hist.Fill(args, ROOT::Experimental::RWeight(0.8));
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
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly and the
   /// \ref Fill(const std::tuple<A...> &args, const V &value) "overload for unweighted filling"
   template <typename... A, typename V>
   void Fill(const std::tuple<A...> &args, const V &value, RWeight weight)
   {
      RValueWeightWrapper wrapper(value, weight.fValue);
      fEngine.Fill(args, wrapper);
      fStats.Fill(std::tuple_cat(args, std::make_tuple(wrapper.fValue)), weight);
   }

   /// Fill an entry into the profile histogram.
   ///
   /// \code
   /// ROOT::Experimental::RProfile profile({/* two dimensions */});
   /// profile.Fill(8.5, 10.5, 23.0);
   /// \endcode
   ///
   /// For weighted filling, pass an RWeight as the last argument:
   /// \code
   /// profile.Fill(8.5, 10.5, 23.0, ROOT::Experimental::RWeight(0.8));
   /// \endcode
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \par See also
   /// the function overloads accepting `std::tuple`
   /// \ref Fill(const std::tuple<A...> &args, const V &value) "for unweighted filling" and
   /// \ref Fill(const std::tuple<A...> &args, const V &value, RWeight weight) "for weighted filling"
   template <typename... A>
   void Fill(const A &...args)
   {
      static_assert(sizeof...(A) >= 2, "need at least two arguments to Fill");
      if constexpr (sizeof...(A) >= 2) {
         auto t = std::forward_as_tuple(args...);
         if constexpr (std::is_same_v<typename Internal::LastType<A...>::type, RWeight>) {
            static constexpr std::size_t N = sizeof...(A) - 2;
            if (N != GetNDimensions()) {
               throw std::invalid_argument("invalid number of arguments to Fill");
            }
            RWeight weight = std::get<N + 1>(t);
            RValueWeightWrapper wrapper(std::get<N>(t), weight.fValue);
            fEngine.FillImpl<N>(t, wrapper);
         } else {
            static constexpr std::size_t N = sizeof...(A) - 1;
            if (N != GetNDimensions()) {
               throw std::invalid_argument("invalid number of arguments to Fill");
            }
            RValueWrapper wrapper(std::get<N>(t));
            fEngine.FillImpl<N>(t, wrapper);
         }
         fStats.Fill(args...);
      }
   }

   /// \}
   /// \name Operations
   /// \{

   /// Add all bin contents and statistics of another profile histogram.
   ///
   /// Throws an exception if the axes configurations are not identical.
   ///
   /// \param[in] other another profile histogram
   void Add(const RProfile &other)
   {
      fEngine.Add(other.fEngine);
      fStats.Add(other.fStats);
   }

   /// Add all bin contents and statistics of another profile histogram using atomic instructions.
   ///
   /// Throws an exception if the axes configurations are not identical.
   ///
   /// \param[in] other another profile histogram that must not be modified during the operation
   void AddAtomic(const RProfile &other)
   {
      fEngine.AddAtomic(other.fEngine);
      fStats.AddAtomic(other.fStats);
   }

   /// Clear all bin contents and statistics.
   void Clear()
   {
      fEngine.Clear();
      fStats.Clear();
   }

   /// Clone this profile histogram.
   ///
   /// Copying all bin contents can be an expensive operation, depending on the number of bins.
   ///
   /// \return the cloned object
   RProfile Clone() const
   {
      RProfile profile(fEngine.Clone());
      profile.fStats = fStats;
      return profile;
   }

   /// Scale all bin contents and statistics.
   ///
   /// \param[in] factor the scale factor
   void Scale(double factor)
   {
      fEngine.Scale(factor);
      fStats.Scale(factor);
   }

   /// \}

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RHist"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
