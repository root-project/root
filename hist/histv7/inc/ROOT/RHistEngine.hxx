/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RHistEngine
#define ROOT_RHistEngine

#include "RAxes.hxx"
#include "RBinIndex.hxx"
#include "RLinearizedIndex.hxx"
#include "RRegularAxis.hxx"

#include <array>
#include <cassert>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
A histogram data structure to bin data along multiple dimensions.

Every call to \ref Fill(const A &... args) "Fill" bins the data according to the axis configuration and increments the
bin content:
\code
ROOT::Experimental::RHistEngine<int> hist(10, 5, 15);
hist.Fill(8.5);
// hist.GetBinContent(ROOT::Experimental::RBinIndex(3)) will return 1
\endcode

The class is templated on the bin content type. For counting, as in the example above, it may be an integer type such as
`int` or `long`. Narrower types such as `unsigned char` or `short` are supported, but may overflow due to their limited
range and must be used with care. For weighted filling, the bin content type must be a floating-point type such as
`float` or `double`. Note that `float` has a limited significant precision of 24 bits.

An object can have arbitrary dimensionality determined at run-time. The axis configuration is passed as a vector of
RAxisVariant:
\code
std::vector<ROOT::Experimental::RAxisVariant> axes;
axes.push_back(ROOT::Experimental::RRegularAxis(10, 5, 15));
axes.push_back(ROOT::Experimental::RVariableBinAxis({1, 10, 100, 1000}));
ROOT::Experimental::RHistEngine<int> hist(axes);
// hist.GetNDimensions() will return 2
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
template <typename BinContentType>
class RHistEngine final {
   /// The axis configuration for this histogram. Relevant methods are forwarded from the public interface.
   Internal::RAxes fAxes;
   /// The bin contents for this histogram
   std::vector<BinContentType> fBinContents;

public:
   /// Construct a histogram engine.
   ///
   /// \param[in] axes the axis objects, must have size > 0
   explicit RHistEngine(std::vector<RAxisVariant> axes) : fAxes(std::move(axes))
   {
      fBinContents.resize(fAxes.ComputeTotalNBins());
   }

   /// Construct a one-dimensional histogram engine with a regular axis.
   ///
   /// \param[in] nNormalBins the number of normal bins, must be > 0
   /// \param[in] low the lower end of the axis interval (inclusive)
   /// \param[in] high the upper end of the axis interval (exclusive), must be > low
   /// \par See also
   /// the \ref RRegularAxis::RRegularAxis(std::size_t nNormalBins, double low, double high, bool enableFlowBins)
   /// "constructor of RRegularAxis"
   RHistEngine(std::size_t nNormalBins, double low, double high) : RHistEngine({RRegularAxis(nNormalBins, low, high)})
   {
   }

   // Copy constructor and assignment operator are deleted to avoid surprises.
   RHistEngine(const RHistEngine<BinContentType> &) = delete;
   RHistEngine(RHistEngine<BinContentType> &&) = default;
   RHistEngine<BinContentType> &operator=(const RHistEngine<BinContentType> &) = delete;
   RHistEngine<BinContentType> &operator=(RHistEngine<BinContentType> &&) = default;
   ~RHistEngine() = default;

   const std::vector<RAxisVariant> &GetAxes() const { return fAxes.Get(); }
   std::size_t GetNDimensions() const { return fAxes.GetNDimensions(); }
   std::size_t GetTotalNBins() const { return fBinContents.size(); }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RHistEngine<int> hist({/* two dimensions */});
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
      // We could rely on RAxes::ComputeGlobalIndex to check the number of arguments, but its exception message might
      // be confusing for users.
      if (N != GetNDimensions()) {
         throw std::invalid_argument("invalid number of indices passed to GetBinContent");
      }
      RLinearizedIndex index = fAxes.ComputeGlobalIndex(indices);
      if (!index.fValid) {
         throw std::invalid_argument("bin not found in GetBinContent");
      }
      assert(index.fIndex < fBinContents.size());
      return fBinContents[index.fIndex];
   }

   /// Get the content of a single bin.
   ///
   /// \code
   /// ROOT::Experimental::RHistEngine<int> hist({/* two dimensions */});
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
      std::array<RBinIndex, sizeof...(A)> indices{args...};
      return GetBinContent(indices);
   }

   /// Fill an entry into the histogram.
   ///
   /// \code
   /// ROOT::Experimental::RHistEngine<int> hist({/* two dimensions */});
   /// auto args = std::make_tuple(8.5, 10.5);
   /// hist.Fill(args);
   /// \endcode
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration.
   ///
   /// \param[in] args the arguments for each axis
   /// \par See also
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly
   template <typename... A>
   void Fill(const std::tuple<A...> &args)
   {
      // We could rely on RAxes::ComputeGlobalIndex to check the number of arguments, but its exception message might
      // be confusing for users.
      if (sizeof...(A) != GetNDimensions()) {
         throw std::invalid_argument("invalid number of arguments to Fill");
      }
      RLinearizedIndex index = fAxes.ComputeGlobalIndex(args);
      if (index.fValid) {
         assert(index.fIndex < fBinContents.size());
         fBinContents[index.fIndex]++;
      }
   }

   /// Fill an entry into the histogram.
   ///
   /// \code
   /// ROOT::Experimental::RHistEngine<int> hist({/* two dimensions */});
   /// hist.Fill(8.5, 10.5);
   /// \endcode
   ///
   /// If one of the arguments is outside the corresponding axis and flow bins are disabled, the entry will be silently
   /// discarded.
   ///
   /// Throws an exception if the number of arguments does not match the axis configuration.
   ///
   /// \param[in] args the arguments for each axis
   /// \par See also
   /// the \ref Fill(const std::tuple<A...> &args) "function overload" accepting `std::tuple`
   template <typename... A>
   void Fill(const A &...args)
   {
      Fill(std::forward_as_tuple(args...));
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RHistEngine"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
