/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RHistStats
#define ROOT_RHistStats

#include "RHistUtils.hxx"
#include "RWeight.hxx"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
Histogram statistics of unbinned values.

Every call to \ref Fill(const A &... args) "Fill" updates sums to compute the number of effective entries as well as the
arithmetic mean and other statistical quantities per dimension:
\code
ROOT::Experimental::RHistStats stats(1);
stats.Fill(8.5);
stats.Fill(1.5);
// stats.ComputeMean() will return 5.0
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RHistStats final {
public:
   /// Statistics for one dimension.
   struct RDimensionStats final {
      double fSumWX = 0.0;
      double fSumWX2 = 0.0;
      double fSumWX3 = 0.0;
      double fSumWX4 = 0.0;

      void Add(double x)
      {
         fSumWX += x;
         fSumWX2 += x * x;
         fSumWX3 += x * x * x;
         fSumWX4 += x * x * x * x;
      }

      void Add(double x, double w)
      {
         fSumWX += w * x;
         fSumWX2 += w * x * x;
         fSumWX3 += w * x * x * x;
         fSumWX4 += w * x * x * x * x;
      }

      void Add(const RDimensionStats &other)
      {
         fSumWX += other.fSumWX;
         fSumWX2 += other.fSumWX2;
         fSumWX3 += other.fSumWX3;
         fSumWX4 += other.fSumWX4;
      }

      /// Add another statistics object using atomic instructions.
      ///
      /// \param[in] other another statistics object that must not be modified during the operation
      void AddAtomic(const RDimensionStats &other)
      {
         Internal::AtomicAdd(&fSumWX, other.fSumWX);
         Internal::AtomicAdd(&fSumWX2, other.fSumWX2);
         Internal::AtomicAdd(&fSumWX3, other.fSumWX3);
         Internal::AtomicAdd(&fSumWX4, other.fSumWX4);
      }

      void Clear()
      {
         fSumWX = 0.0;
         fSumWX2 = 0.0;
         fSumWX3 = 0.0;
         fSumWX4 = 0.0;
      }

      void Scale(double factor)
      {
         fSumWX *= factor;
         fSumWX2 *= factor;
         fSumWX3 *= factor;
         fSumWX4 *= factor;
      }
   };

private:
   /// The number of entries
   std::uint64_t fNEntries = 0;
   /// The sum of weights
   double fSumW = 0.0;
   /// The sum of weights squared
   double fSumW2 = 0.0;
   /// The sums per dimension
   std::vector<RDimensionStats> fDimensionStats;

public:
   /// Construct a statistics object.
   ///
   /// \param[in] nDimensions the number of dimensions, must be > 0
   explicit RHistStats(std::size_t nDimensions)
   {
      if (nDimensions == 0) {
         throw std::invalid_argument("nDimensions must be > 0");
      }
      fDimensionStats.resize(nDimensions);
   }

   std::size_t GetNDimensions() const { return fDimensionStats.size(); }

   std::uint64_t GetNEntries() const { return fNEntries; }
   double GetSumW() const { return fSumW; }
   double GetSumW2() const { return fSumW2; }

   const RDimensionStats &GetDimensionStats(std::size_t dim = 0) const { return fDimensionStats.at(dim); }

   /// Add all entries from another statistics object.
   ///
   /// Throws an exception if the number of dimensions are not identical.
   ///
   /// \param[in] other another statistics object
   void Add(const RHistStats &other)
   {
      if (fDimensionStats.size() != other.fDimensionStats.size()) {
         throw std::invalid_argument("number of dimensions not identical in Add");
      }
      fNEntries += other.fNEntries;
      fSumW += other.fSumW;
      fSumW2 += other.fSumW2;
      for (std::size_t i = 0; i < fDimensionStats.size(); i++) {
         fDimensionStats[i].Add(other.fDimensionStats[i]);
      }
   }

   /// Add all entries from another statistics object using atomic instructions.
   ///
   /// Throws an exception if the number of dimensions are not identical.
   ///
   /// \param[in] other another statistics object that must not be modified during the operation
   void AddAtomic(const RHistStats &other)
   {
      if (fDimensionStats.size() != other.fDimensionStats.size()) {
         throw std::invalid_argument("number of dimensions not identical in Add");
      }
      Internal::AtomicAdd(&fNEntries, other.fNEntries);
      Internal::AtomicAdd(&fSumW, other.fSumW);
      Internal::AtomicAdd(&fSumW2, other.fSumW2);
      for (std::size_t i = 0; i < fDimensionStats.size(); i++) {
         fDimensionStats[i].AddAtomic(other.fDimensionStats[i]);
      }
   }

   /// Clear this statistics object.
   void Clear()
   {
      fNEntries = 0;
      fSumW = 0;
      fSumW2 = 0;
      for (std::size_t i = 0; i < fDimensionStats.size(); i++) {
         fDimensionStats[i].Clear();
      }
   }

   /// Compute the number of effective entries.
   ///
   /// \f[
   /// \frac{(\sum w_i)^2}{\sum w_i^2}
   /// \f]
   ///
   /// \return the number of effective entries
   double ComputeNEffectiveEntries() const
   {
      if (fSumW2 == 0) {
         return 0;
      }
      return fSumW * fSumW / fSumW2;
   }

   /// Compute the arithmetic mean of unbinned values.
   ///
   /// \f[
   /// \mu = \frac{\sum w_i \cdot x_i}{\sum w_i}
   /// \f]
   ///
   /// \param[in] dim the dimension index, starting at 0
   /// \return the arithmetic mean of unbinned values
   double ComputeMean(std::size_t dim = 0) const
   {
      // First get the statistics, which includes checking the argument.
      auto &stats = fDimensionStats.at(dim);
      if (fSumW == 0) {
         return 0;
      }
      return stats.fSumWX / fSumW;
   }

   /// Compute the variance of unbinned values.
   ///
   /// This function computes the uncorrected sample variance:
   /// \f[
   /// \sigma^2 = \frac{1}{\sum w_i} \sum(w_i \cdot x_i - \mu)^2
   /// \f]
   /// With some rewriting, this is equivalent to:
   /// \f[
   /// \sigma^2 = \frac{\sum w_i \cdot x_i^2}{\sum w_i} - \mu^2
   /// \f]
   ///
   /// This function does not include Bessel's correction needed for an unbiased estimator of population variance.
   /// In other words, the return value is a biased estimation lower than the actual population variance.
   ///
   /// \param[in] dim the dimension index, starting at 0
   /// \return the variance of unbinned values
   double ComputeVariance(std::size_t dim = 0) const
   {
      // First get the statistics, which includes checking the argument.
      auto &stats = fDimensionStats.at(dim);
      if (fSumW == 0) {
         return 0;
      }
      double mean = ComputeMean(dim);
      return stats.fSumWX2 / fSumW - mean * mean;
   }

   /// Compute the standard deviation of unbinned values.
   ///
   /// This function computes the uncorrected sample standard deviation:
   /// \f[
   /// \sigma = \sqrt{\frac{1}{\sum w_i} \sum(w_i \cdot x_i - \mu)^2}
   /// \f]
   /// With some rewriting, this is equivalent to:
   /// \f[
   /// \sigma = \sqrt{\frac{\sum w_i \cdot x_i^2}{\sum w_i} - \frac{(\sum w_i \cdot x_i)^2}{(\sum w_i)^2}}
   /// \f]
   ///
   /// This function does not include Bessel's correction needed for an unbiased estimator of population variance.
   /// In other words, the return value is a biased estimation lower than the actual population standard deviation.
   ///
   /// \param[in] dim the dimension index, starting at 0
   /// \return the standard deviation of unbinned values
   double ComputeStdDev(std::size_t dim = 0) const { return std::sqrt(ComputeVariance(dim)); }

   // clang-format off
   /// Compute the skewness of unbinned values.
   ///
   /// The skewness is the third standardized moment:
   /// \f[
   /// E\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]
   /// \f]
   /// With support for weighted filling and after some rewriting, it is computed as:
   /// \f[
   /// \frac{\frac{\sum w_i \cdot x_i^3}{\sum w_i} - 3 \cdot \frac{\sum w_i \cdot x_i^2}{\sum w_i} \cdot \mu + 2 \cdot \mu^3}{\sigma^3}
   /// \f]
   ///
   /// \param[in] dim the dimension index, starting at 0
   /// \return the skewness of unbinned values
   // clang-format on
   double ComputeSkewness(std::size_t dim = 0) const
   {
      // First get the statistics, which includes checking the argument.
      auto &stats = fDimensionStats.at(dim);
      if (fSumW == 0) {
         return 0;
      }
      double mean = ComputeMean(dim);
      double var = ComputeVariance(dim);
      if (var == 0) {
         return 0;
      }
      double EWX3 = stats.fSumWX3 / fSumW;
      double EWX2 = stats.fSumWX2 / fSumW;
      return (EWX3 - 3 * EWX2 * mean + 2 * mean * mean * mean) / std::pow(var, 1.5);
   }

   // clang-format off
   /// Compute the (excess) kurtosis of unbinned values.
   ///
   /// The kurtosis is based on the fourth standardized moment:
   /// \f[
   /// E\left[\left(\frac{X - \mu}{\sigma}\right)^4\right]
   /// \f]
   /// The excess kurtosis subtracts 3 from the standardized moment to have a value of 0 for a normal distribution:
   /// \f[
   /// E\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3
   /// \f]
   ///
   /// With support for weighted filling and after some rewriting, the (excess kurtosis) is computed as:
   /// \f[
   /// \frac{\frac{\sum w_i \cdot x_i^4}{\sum w_i} - 4 \cdot \frac{\sum w_i \cdot x_i^3}{\sum w_i} \cdot \mu + 6 \cdot \frac{\sum w_i \cdot x_i^2}{\sum w_i} \cdot \mu^2 - 3 \cdot \mu^4}{\sigma^4} - 3
   /// \f]
   ///
   /// \param[in] dim the dimension index, starting at 0
   /// \return the (excess) kurtosis of unbinned values
   // clang-format on
   double ComputeKurtosis(std::size_t dim = 0) const
   {
      // First get the statistics, which includes checking the argument.
      auto &stats = fDimensionStats.at(dim);
      if (fSumW == 0) {
         return 0;
      }
      double mean = ComputeMean(dim);
      double var = ComputeVariance(dim);
      if (var == 0) {
         return 0;
      }
      double EWX4 = stats.fSumWX4 / fSumW;
      double EWX3 = stats.fSumWX3 / fSumW;
      double EWX2 = stats.fSumWX2 / fSumW;
      return (EWX4 - 4 * EWX3 * mean + 6 * EWX2 * mean * mean - 3 * mean * mean * mean * mean) / (var * var) - 3;
   }

private:
   template <std::size_t I, typename... A>
   void FillImpl(const std::tuple<A...> &args)
   {
      fDimensionStats[I].Add(std::get<I>(args));
      if constexpr (I + 1 < sizeof...(A)) {
         FillImpl<I + 1>(args);
      }
   }

   template <std::size_t I, std::size_t N, typename... A>
   void FillImpl(const std::tuple<A...> &args, double w)
   {
      fDimensionStats[I].Add(std::get<I>(args), w);
      if constexpr (I + 1 < N) {
         FillImpl<I + 1, N>(args, w);
      }
   }

public:
   /// Fill an entry into this statistics object.
   ///
   /// \code
   /// ROOT::Experimental::RHistStats stats(2);
   /// auto args = std::make_tuple(8.5, 10.5);
   /// stats.Fill(args);
   /// \endcode
   ///
   /// Throws an exception if the number of arguments does not match the number of dimensions.
   ///
   /// \param[in] args the arguments for each dimension
   /// \par See also
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly and the
   /// \ref Fill(const std::tuple<A...> &args, RWeight weight) "overload for weighted filling"
   template <typename... A>
   void Fill(const std::tuple<A...> &args)
   {
      if (sizeof...(A) != fDimensionStats.size()) {
         throw std::invalid_argument("invalid number of arguments to Fill");
      }
      fNEntries++;
      fSumW++;
      fSumW2++;
      FillImpl<0>(args);
   }

   /// Fill an entry into this statistics object with a weight.
   ///
   /// \code
   /// ROOT::Experimental::RHistStats stats(2);
   /// auto args = std::make_tuple(8.5, 10.5);
   /// stats.Fill(args, ROOT::Experimental::RWeight(0.8));
   /// \endcode
   ///
   /// \param[in] args the arguments for each dimension
   /// \param[in] weight the weight for this entry
   /// \par See also
   /// the \ref Fill(const A &... args) "variadic function template overload" accepting arguments directly and the
   /// \ref Fill(const std::tuple<A...> &args) "overload for unweighted filling"
   template <typename... A>
   void Fill(const std::tuple<A...> &args, RWeight weight)
   {
      if (sizeof...(A) != fDimensionStats.size()) {
         throw std::invalid_argument("invalid number of arguments to Fill");
      }
      fNEntries++;
      double w = weight.fValue;
      fSumW += w;
      fSumW2 += w * w;
      FillImpl<0, sizeof...(A)>(args, w);
   }

   /// Fill an entry into this statistics object.
   ///
   /// \code
   /// ROOT::Experimental::RHistStats stats(2);
   /// stats.Fill(8.5, 10.5);
   /// \endcode
   /// For weighted filling, pass an RWeight as the last argument:
   /// \code
   /// ROOT::Experimental::RHistStats stats(2);
   /// stats.Fill(8.5, 10.5, ROOT::Experimental::RWeight(0.8));
   /// \endcode
   ///
   /// Throws an exception if the number of arguments does not match the number of dimensions.
   ///
   /// \param[in] args the arguments for each dimension
   /// \par See also
   /// the function overloads accepting `std::tuple` \ref Fill(const std::tuple<A...> &args) "for unweighted filling"
   /// and \ref Fill(const std::tuple<A...> &args, RWeight) "for weighted filling"
   template <typename... A>
   void Fill(const A &...args)
   {
      auto t = std::forward_as_tuple(args...);
      if constexpr (std::is_same_v<typename Internal::LastType<A...>::type, RWeight>) {
         static constexpr std::size_t N = sizeof...(A) - 1;
         if (N != fDimensionStats.size()) {
            throw std::invalid_argument("invalid number of arguments to Fill");
         }
         fNEntries++;
         double w = std::get<N>(t).fValue;
         fSumW += w;
         fSumW2 += w * w;
         FillImpl<0, N>(t, w);
      } else {
         Fill(t);
      }
   }

   /// Scale the histogram statistics.
   ///
   /// \param[in] factor the scale factor
   void Scale(double factor)
   {
      fSumW *= factor;
      fSumW2 *= factor * factor;
      for (std::size_t i = 0; i < fDimensionStats.size(); i++) {
         fDimensionStats[i].Scale(factor);
      }
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RHistStats"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
