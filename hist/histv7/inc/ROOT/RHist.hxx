/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RHist
#define ROOT_RHist

#include "RAxes.hxx" // for RAxisVariant
#include "RBinIndex.hxx"
#include "RHistEngine.hxx"
#include "RHistStats.hxx"
#include "RRegularAxis.hxx"
#include "RWeight.hxx"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

// forward declaration for friend declaration
template <typename BinContentType>
class RHistFillContext;

/**
A histogram for aggregation of data along multiple dimensions.

Every call to \ref Fill(const A &... args) "Fill" increments the bin content and is reflected in global statistics:
\code
ROOT::Experimental::RHist<int> hist(10, {5, 15});
hist.Fill(8.5);
// hist.GetBinContent(ROOT::Experimental::RBinIndex(3)) will return 1
\endcode

The class is templated on the bin content type. For counting, as in the example above, it may be an integral type such
as `int` or `long`. Narrower types such as `unsigned char` or `short` are supported, but may overflow due to their
limited range and must be used with care. For weighted filling, the bin content type must not be an integral type, but
a floating-point type such as `float` or `double`, or the special type RBinWithError. Note that `float` has a limited
significand precision of 24 bits.

An object can have arbitrary dimensionality determined at run-time. The axis configuration is passed as a vector of
RAxisVariant:
\code
std::vector<ROOT::Experimental::RAxisVariant> axes;
axes.push_back(ROOT::Experimental::RRegularAxis(10, 5, 15));
axes.push_back(ROOT::Experimental::RVariableBinAxis({1, 10, 100, 1000}));
ROOT::Experimental::RHist<int> hist(axes);
// hist.GetNDimensions() will return 2
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
template <typename BinContentType>
class RHist final {
   friend class RHistFillContext<BinContentType>;

   /// The histogram engine including the bin contents.
   RHistEngine<BinContentType> fEngine;
   /// The global histogram statistics.
   RHistStats fStats;

   /// Private constructor based off an engine.
   RHist(RHistEngine<BinContentType> engine) : fEngine(std::move(engine)), fStats(fEngine.GetNDimensions()) {}

public:
   /// Construct a histogram.
   ///
   /// \param[in] axes the axis objects, must have size > 0
   explicit RHist(std::vector<RAxisVariant> axes) : fEngine(std::move(axes)), fStats(fEngine.GetNDimensions()) {}

   /// Construct a one-dimensional histogram engine with a regular axis.
   ///
   /// \param[in] nNormalBins the number of normal bins, must be > 0
   /// \param[in] interval the axis interval (lower end inclusive, upper end exclusive)
   /// \par See also
   /// the \ref RRegularAxis::RRegularAxis(std::uint64_t nNormalBins, std::pair<double, double> interval, bool
   /// enableFlowBins) "constructor of RRegularAxis"
   RHist(std::uint64_t nNormalBins, std::pair<double, double> interval) : RHist({RRegularAxis(nNormalBins, interval)})
   {
   }

   /// The copy constructor is deleted.
   ///
   /// Copying all bin contents can be an expensive operation, depending on the number of bins. If required, users can
   /// explicitly call Clone().
   RHist(const RHist &) = delete;
   /// Efficiently move construct a histogram.
   ///
   /// After this operation, the moved-from object is invalid.
   RHist(RHist &&) = default;

   /// The copy assignment operator is deleted.
   ///
   /// Copying all bin contents can be an expensive operation, depending on the number of bins. If required, users can
   /// explicitly call Clone().
   RHist &operator=(const RHist &) = delete;
   /// Efficiently move a histogram.
   ///
   /// After this operation, the moved-from object is invalid.
   RHist &operator=(RHist &&) = default;

   ~RHist() = default;

   const RHistEngine<BinContentType> &GetEngine() const { return fEngine; }
   const RHistStats &GetStats() const { return fStats; }

   const std::vector<RAxisVariant> &GetAxes() const { return fEngine.GetAxes(); }
   std::size_t GetNDimensions() const { return fEngine.GetNDimensions(); }
   std::uint64_t GetTotalNBins() const { return fEngine.GetTotalNBins(); }

   std::uint64_t GetNEntries() const { return fStats.GetNEntries(); }
   /// \copydoc RHistStats::ComputeNEffectiveEntries()
   double ComputeNEffectiveEntries() const { return fStats.ComputeNEffectiveEntries(); }
   /// \copydoc RHistStats::ComputeMean()
   double ComputeMean(std::size_t dim = 0) const { return fStats.ComputeMean(dim); }
   /// \copydoc RHistStats::ComputeStdDev()
   double ComputeStdDev(std::size_t dim = 0) const { return fStats.ComputeStdDev(dim); }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RHist<int> hist({/* two dimensions */});
   /// std::array<ROOT::Experimental::RBinIndex, 2> indices = {3, 5};
   /// int content = hist.GetBinContent(indices);
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
   const BinContentType &GetBinContent(const std::array<RBinIndex, N> &indices) const
   {
      return fEngine.GetBinContent(indices);
   }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RHist<int> hist({/* two dimensions */});
   /// int content = hist.GetBinContent(ROOT::Experimental::RBinIndex(3), ROOT::Experimental::RBinIndex(5));
   /// // ... or construct the RBinIndex arguments implicitly from integers:
   /// content = hist.GetBinContent(3, 5);
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
   /// the \ref GetBinContent(const std::array<RBinIndex, N> &indices) const "function overload" accepting
   /// `std::array`
   template <typename... A>
   const BinContentType &GetBinContent(const A &...args) const
   {
      return fEngine.GetBinContent(args...);
   }

   /// Add all bin contents and statistics of another histogram.
   ///
   /// Throws an exception if the axes configurations are not identical.
   ///
   /// \param[in] other another histogram
   void Add(const RHist &other)
   {
      fEngine.Add(other.fEngine);
      fStats.Add(other.fStats);
   }

   /// Add all bin contents and statistics of another histogram using atomic instructions.
   ///
   /// Throws an exception if the axes configurations are not identical.
   ///
   /// \param[in] other another histogram that must not be modified during the operation
   void AddAtomic(const RHist &other)
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

   /// Clone this histogram.
   ///
   /// Copying all bin contents can be an expensive operation, depending on the number of bins.
   ///
   /// \return the cloned object
   RHist Clone() const
   {
      RHist h(fEngine.Clone());
      h.fStats = fStats;
      return h;
   }

   /// Fill an entry into the histogram.
   ///
   /// \code
   /// ROOT::Experimental::RHist<int> hist({/* two dimensions */});
   /// auto args = std::make_tuple(8.5, 10.5);
   /// hist.Fill(args);
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
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly and the
   /// \ref Fill(const std::tuple<A...> &args, RWeight weight) "overload for weighted filling"
   template <typename... A>
   void Fill(const std::tuple<A...> &args)
   {
      fEngine.Fill(args);
      fStats.Fill(args);
   }

   /// Fill an entry into the histogram with a weight.
   ///
   /// This overload is not available for integral bin content types (see \ref RHistEngine::SupportsWeightedFilling).
   ///
   /// \code
   /// ROOT::Experimental::RHist<float> hist({/* two dimensions */});
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
   /// \param[in] weight the weight for this entry
   /// \par See also
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly and the
   /// \ref Fill(const std::tuple<A...> &args) "overload for unweighted filling"
   template <typename... A>
   void Fill(const std::tuple<A...> &args, RWeight weight)
   {
      fEngine.Fill(args, weight);
      fStats.Fill(args, weight);
   }

   /// Fill an entry into the histogram.
   ///
   /// \code
   /// ROOT::Experimental::RHist<int> hist({/* two dimensions */});
   /// hist.Fill(8.5, 10.5);
   /// \endcode
   ///
   /// For weighted filling, pass an RWeight as the last argument:
   /// \code
   /// ROOT::Experimental::RHist<float> hist({/* two dimensions */});
   /// hist.Fill(8.5, 10.5, ROOT::Experimental::RWeight(0.8));
   /// \endcode
   /// This is not available for integral bin content types (see \ref RHistEngine::SupportsWeightedFilling).
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration, or if an argument cannot be
   /// converted for the axis type at run-time.
   ///
   /// \param[in] args the arguments for each axis
   /// \par See also
   /// the function overloads accepting `std::tuple` \ref Fill(const std::tuple<A...> &args) "for unweighted filling"
   /// and \ref Fill(const std::tuple<A...> &args, RWeight) "for weighted filling"
   template <typename... A>
   void Fill(const A &...args)
   {
      fEngine.Fill(args...);
      fStats.Fill(args...);
   }

   /// Scale all histogram bin contents and statistics.
   ///
   /// This method is not available for integral bin content types.
   ///
   /// \param[in] factor the scale factor
   void Scale(double factor)
   {
      fEngine.Scale(factor);
      fStats.Scale(factor);
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RHist"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
