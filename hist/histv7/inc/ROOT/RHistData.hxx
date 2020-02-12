/// \file ROOT/RHistData.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-06-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistData
#define ROOT7_RHistData

#include <cmath>
#include <vector>
#include "ROOT/RSpan.hxx"
#include "ROOT/RHistUtils.hxx"

namespace ROOT {
namespace Experimental {

template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
class RHist;

/**
 \class RHistStatContent
 Basic histogram statistics, keeping track of the bin content and the total
 number of calls to Fill().
 */
template <int DIMENSIONS, class PRECISION>
class RHistStatContent {
public:
   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;
   /// Type of the bin content array.
   using Content_t = std::vector<PRECISION>;

   /**
    \class RConstBinStat
    Const view on a RHistStatContent for a given bin.
   */
   class RConstBinStat {
   public:
      RConstBinStat(const RHistStatContent &stat, int index): fContent(stat.GetBinContent(index)) {}
      PRECISION GetContent() const { return fContent; }

   private:
      PRECISION fContent; ///< The content of this bin.
   };

   /**
    \class RBinStat
    Modifying view on a RHistStatContent for a given bin.
   */
   class RBinStat {
   public:
      RBinStat(RHistStatContent &stat, int index): fContent(stat.GetBinContent(index)) {}
      PRECISION &GetContent() const { return fContent; }

   private:
      PRECISION &fContent; ///< The content of this bin.
   };

   using ConstBinStat_t = RConstBinStat;
   using BinStat_t = RBinStat;

private:
   /// Number of calls to Fill().
   int64_t fEntries = 0;

   /// Bin content.
   Content_t fBinContent;

public:
   RHistStatContent() = default;
   RHistStatContent(size_t in_size): fBinContent(in_size) {}

   /// Add weight to the bin content at binidx.
   void Fill(const CoordArray_t & /*x*/, int binidx, Weight_t weight = 1.)
   {
      fBinContent[binidx] += weight;
      ++fEntries;
   }

   /// Get the number of entries filled into the histogram - i.e. the number of
   /// calls to Fill().
   int64_t GetEntries() const { return fEntries; }

   /// Get the number of bins.
   size_t size() const noexcept { return fBinContent.size(); }

   /// Get the bin content for the given bin.
   Weight_t operator[](int idx) const { return fBinContent[idx]; }
   /// Get the bin content for the given bin (non-const).
   Weight_t &operator[](int idx) { return fBinContent[idx]; }

   /// Get the bin content for the given bin.
   Weight_t GetBinContent(int idx) const { return fBinContent[idx]; }
   /// Get the bin content for the given bin (non-const).
   Weight_t &GetBinContent(int idx) { return fBinContent[idx]; }

   /// Retrieve the content array.
   const Content_t &GetContentArray() const { return fBinContent; }
   /// Retrieve the content array (non-const).
   Content_t &GetContentArray() { return fBinContent; }

   /// Merge with other RHistStatContent, assuming same bin configuration
   void Add(const RHistStatContent& other) {
      assert(fBinContent.size() == other.fBinContent.size());
      fEntries += other.fEntries;
      for (size_t b = 0; b < fBinContent.size(); ++b)
         fBinContent[b] += other.fBinContent[b];
   }
};

/**
 \class RHistStatTotalSumOfWeights
 Keeps track of the histogram's total sum of weights.
 */
template <int DIMENSIONS, class PRECISION>
class RHistStatTotalSumOfWeights {
public:
   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;

   /**
    \class RBinStat
    No-op; this class does not provide per-bin statistics.
   */
   class RBinStat {
   public:
      RBinStat(const RHistStatTotalSumOfWeights &, int) {}
   };

   using ConstBinStat_t = RBinStat;
   using BinStat_t = RBinStat;

private:
   /// Sum of weights.
   PRECISION fSumWeights = 0;

public:
   RHistStatTotalSumOfWeights() = default;
   RHistStatTotalSumOfWeights(size_t) {}

   /// Add weight to the bin content at binidx.
   void Fill(const CoordArray_t & /*x*/, int, Weight_t weight = 1.) { fSumWeights += weight; }

   /// Get the sum of weights.
   Weight_t GetSumOfWeights() const { return fSumWeights; }

   /// Merge with other RHistStatTotalSumOfWeights data, assuming same bin configuration
   void Add(const RHistStatTotalSumOfWeights& other) {
      fSumWeights += other.fSumWeights;
   }
};

/**
 \class RHistStatTotalSumOfSquaredWeights
 Keeps track of the histogram's total sum of squared weights.
 */
template <int DIMENSIONS, class PRECISION>
class RHistStatTotalSumOfSquaredWeights {
public:
   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;

   /**
    \class RBinStat
    No-op; this class does not provide per-bin statistics.
   */
   class RBinStat {
   public:
      RBinStat(const RHistStatTotalSumOfSquaredWeights &, int) {}
   };

   using ConstBinStat_t = RBinStat;
   using BinStat_t = RBinStat;

private:
   /// Sum of (weights^2).
   PRECISION fSumWeights2 = 0;

public:
   RHistStatTotalSumOfSquaredWeights() = default;
   RHistStatTotalSumOfSquaredWeights(size_t) {}

   /// Add weight to the bin content at binidx.
   void Fill(const CoordArray_t & /*x*/, int /*binidx*/, Weight_t weight = 1.) { fSumWeights2 += weight * weight; }

   /// Get the sum of weights.
   Weight_t GetSumOfSquaredWeights() const { return fSumWeights2; }

   /// Merge with other RHistStatTotalSumOfSquaredWeights data, assuming same bin configuration
   void Add(const RHistStatTotalSumOfSquaredWeights& other) {
      fSumWeights2 += other.fSumWeights2;
   }
};

/**
 \class RHistStatUncertainty
 Histogram statistics to keep track of the Poisson uncertainty per bin.
 */
template <int DIMENSIONS, class PRECISION>
class RHistStatUncertainty {

public:
   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;
   /// Type of the bin content array.
   using Content_t = std::vector<PRECISION>;

   /**
    \class RConstBinStat
    Const view on a RHistStatUncertainty for a given bin.
   */
   class RConstBinStat {
   public:
      RConstBinStat(const RHistStatUncertainty &stat, int index): fSumW2(stat.GetSumOfSquaredWeights(index)) {}
      PRECISION GetSumW2() const { return fSumW2; }

      double GetUncertaintyImpl() const { return std::sqrt(std::abs(fSumW2)); }

   private:
      PRECISION fSumW2; ///< The bin's sum of square of weights.
   };

   /**
    \class RBinStat
    Modifying view on a RHistStatUncertainty for a given bin.
   */
   class RBinStat {
   public:
      RBinStat(RHistStatUncertainty &stat, int index): fSumW2(stat.GetSumOfSquaredWeights(index)) {}
      PRECISION &GetSumW2() const { return fSumW2; }
      // Can never modify this. Set GetSumW2() instead.
      double GetUncertaintyImpl() const { return std::sqrt(std::abs(fSumW2)); }

   private:
      PRECISION &fSumW2; ///< The bin's sum of square of weights.
   };

   using ConstBinStat_t = RConstBinStat;
   using BinStat_t = RBinStat;

private:
   /// Uncertainty of the content for each bin.
   Content_t fSumWeightsSquared; ///< Sum of squared weights

public:
   RHistStatUncertainty() = default;
   RHistStatUncertainty(size_t size): fSumWeightsSquared(size) {}

   /// Add weight to the bin at binidx; the coordinate was x.
   void Fill(const CoordArray_t & /*x*/, int binidx, Weight_t weight = 1.)
   {
      fSumWeightsSquared[binidx] += weight * weight;
   }

   /// Calculate a bin's (Poisson) uncertainty of the bin content as the
   /// square-root of the bin's sum of squared weights.
   double GetBinUncertaintyImpl(int binidx) const { return std::sqrt(fSumWeightsSquared[binidx]); }

   /// Get a bin's sum of squared weights.
   Weight_t GetSumOfSquaredWeights(int binidx) const { return fSumWeightsSquared[binidx]; }

   /// Get a bin's sum of squared weights.
   Weight_t &GetSumOfSquaredWeights(int binidx) { return fSumWeightsSquared[binidx]; }

   /// Get the structure holding the sum of squares of weights.
   const std::vector<double> &GetSumOfSquaredWeights() const { return fSumWeightsSquared; }
   /// Get the structure holding the sum of squares of weights (non-const).
   std::vector<double> &GetSumOfSquaredWeights() { return fSumWeightsSquared; }

   /// Merge with other RHistStatUncertainty data, assuming same bin configuration
   void Add(const RHistStatUncertainty& other) {
      assert(fSumWeightsSquared.size() == other.fSumWeightsSquared.size());
      for (size_t b = 0; b < fSumWeightsSquared.size(); ++b)
         fSumWeightsSquared[b] += other.fSumWeightsSquared[b];
   }
};

/** \class RHistDataMomentUncert
  For now do as RH1: calculate first (xw) and second (x^2w) moment.
*/
template <int DIMENSIONS, class PRECISION>
class RHistDataMomentUncert {
public:
   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;
   /// Type of the bin content array.
   using Content_t = std::vector<PRECISION>;

   /**
    \class RBinStat
    No-op; this class does not provide per-bin statistics.
   */
   class RBinStat {
   public:
      RBinStat(const RHistDataMomentUncert &, int) {}
   };

   using ConstBinStat_t = RBinStat;
   using BinStat_t = RBinStat;

private:
   std::array<Weight_t, DIMENSIONS> fMomentXW;
   std::array<Weight_t, DIMENSIONS> fMomentX2W;
   // FIXME: Add sum(w.x.y)-style stats.

public:
   RHistDataMomentUncert() = default;
   RHistDataMomentUncert(size_t) {}

   /// Add weight to the bin at binidx; the coordinate was x.
   void Fill(const CoordArray_t &x, int /*binidx*/, Weight_t weight = 1.)
   {
      for (int idim = 0; idim < DIMENSIONS; ++idim) {
         const PRECISION xw = x[idim] * weight;
         fMomentXW[idim] += xw;
         fMomentX2W[idim] += x[idim] * xw;
      }
   }

   // FIXME: Add a way to query the inner data

   /// Merge with other RHistDataMomentUncert data, assuming same bin configuration
   void Add(const RHistDataMomentUncert& other) {
      for (size_t d = 0; d < DIMENSIONS; ++d) {
         fMomentXW[d] += other.fMomentXW[d];
         fMomentX2W[d] += other.fMomentX2W[d];
      }
   }
};

/** \class RHistStatRuntime
  Interface implementing a pure virtual functions DoFill(), DoFillN().
  */
template <int DIMENSIONS, class PRECISION>
class RHistStatRuntime {
public:
   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;
   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;
   /// Type of the bin content array.
   using Content_t = std::vector<PRECISION>;

   /**
    \class RBinStat
    No-op; this class does not provide per-bin statistics.
   */
   class RBinStat {
   public:
      RBinStat(const RHistStatRuntime &, int) {}
   };

   using ConstBinStat_t = RBinStat;
   using BinStat_t = RBinStat;

   RHistStatRuntime() = default;
   RHistStatRuntime(size_t) {}
   virtual ~RHistStatRuntime() = default;

   virtual void DoFill(const CoordArray_t &x, int binidx, Weight_t weightN) = 0;
   void Fill(const CoordArray_t &x, int binidx, Weight_t weight = 1.) { DoFill(x, binidx, weight); }
};

namespace Detail {

/** \class RHistBinStat
  Const view on a bin's statistical data. Combines all STATs' BinStat_t views.
  */
template <class DATA, class... BASES>
class RHistBinStat: public BASES... {
private:
   /// Check whether `double T::GetBinUncertaintyImpl(int)` can be called.
   template <class T>
   static auto HaveUncertainty(const T *This) -> decltype(This->GetUncertaintyImpl());
   /// Fall-back case for check whether `double T::GetBinUncertaintyImpl(int)` can be called.
   template <class T>
   static char HaveUncertainty(...);

public:
   RHistBinStat(DATA &data, int index): BASES(data, index)... {}

   /// Whether this provides storage for uncertainties, or whether uncertainties
   /// are determined as poisson uncertainty of the content.
   static constexpr bool HasBinUncertainty()
   {
      struct AllYourBaseAreBelongToUs: public BASES... {
      };
      return sizeof(HaveUncertainty<AllYourBaseAreBelongToUs>(nullptr)) == sizeof(double);
   }
   /// Calculate the bin content's uncertainty for the given bin, using base class information,
   /// i.e. forwarding to a base's `GetUncertaintyImpl()`.
   template <bool B = true, class = typename std::enable_if<B && HasBinUncertainty()>::type>
   double GetUncertainty() const
   {
      return this->GetUncertaintyImpl();
   }
   /// Calculate the bin content's uncertainty for the given bin, using Poisson
   /// statistics on the absolute bin content. Only available if no base provides
   /// this functionality. Requires GetContent().
   template <bool B = true, class = typename std::enable_if<B && !HasBinUncertainty()>::type>
   double GetUncertainty(...) const
   {
      auto content = this->GetContent();
      return std::sqrt(std::fabs(content));
   }
};

/** \class RHistData
  A RHistImplBase's data, provides accessors to all its statistics.
  */
template <int DIMENSIONS, class PRECISION, class STORAGE, template <int D_, class P_> class... STAT>
class RHistData: public STAT<DIMENSIONS, PRECISION>... {
private:
   /// Check whether `double T::GetBinUncertaintyImpl(int)` can be called.
   template <class T>
   static auto HaveUncertainty(const T *This) -> decltype(This->GetBinUncertaintyImpl(12));
   /// Fall-back case for check whether `double T::GetBinUncertaintyImpl(int)` can be called.
   template <class T>
   static char HaveUncertainty(...);

public:
   /// Matching RHist
   using Hist_t = RHist<DIMENSIONS, PRECISION, STAT...>;

   /// The type of the weight and the bin content.
   using Weight_t = PRECISION;

   /// The type of a (possibly multi-dimensional) coordinate.
   using CoordArray_t = Hist::CoordArray_t<DIMENSIONS>;

   /// The type of a non-modifying view on a bin.
   using ConstHistBinStat_t =
      RHistBinStat<const RHistData, typename STAT<DIMENSIONS, PRECISION>::ConstBinStat_t...>;

   /// The type of a modifying view on a bin.
   using HistBinStat_t = RHistBinStat<RHistData, typename STAT<DIMENSIONS, PRECISION>::BinStat_t...>;

   /// Number of dimensions of the coordinates
   static constexpr int GetNDim() noexcept { return DIMENSIONS; }

   RHistData() = default;

   /// Constructor providing the number of bins (incl under, overflow) to the
   /// base classes.
   RHistData(size_t size): STAT<DIMENSIONS, PRECISION>(size)... {}

   /// Fill weight at x to the bin content at binidx.
   void Fill(const CoordArray_t &x, int binidx, Weight_t weight = 1.)
   {
      // Call Fill() on all base classes.
      // This combines a couple of C++ spells:
      // - "STAT": is a template parameter pack of template template arguments. It
      //           has multiple (or one or no) elements; each is a template name
      //           that needs to be instantiated before it can be used.
      // - "...":  template parameter pack expansion; the expression is evaluated
      //           for each STAT. The expression is
      //           (STAT<DIMENSIONS, PRECISION>::Fill(x, binidx, weight), 0)
      // - "trigger_base_fill{}":
      //           initialization, provides a context in which template parameter
      //           pack expansion happens.
      // - ", 0":  because Fill() returns void it cannot be used as initializer
      //           expression. The trailing ", 0" gives it the type of the trailing
      //           comma-separated expression - int.
      using trigger_base_fill = int[];
      (void)trigger_base_fill{(STAT<DIMENSIONS, PRECISION>::Fill(x, binidx, weight), 0)...};
   }

   /// Merge with other statistics of the same type.
   ///
   /// The implementation assumes that the number of bins is equal.
   void Add(const RHistData &other)
   {
      // Call Add() on all base classes, using the same tricks as Fill().
      using trigger_base_add = int[];
      (void)trigger_base_add{(STAT<DIMENSIONS, PRECISION>::Add(other), 0)...};
   }

   /// Whether this provides storage for uncertainties, or whether uncertainties
   /// are determined as poisson uncertainty of the content.
   static constexpr bool HasBinUncertainty()
   {
      struct AllYourBaseAreBelongToUs: public STAT<DIMENSIONS, PRECISION>... {
      };
      return sizeof(HaveUncertainty<AllYourBaseAreBelongToUs>(nullptr)) == sizeof(double);
   }

   /// Calculate the bin content's uncertainty for the given bin, using base class information,
   /// i.e. forwarding to a base's `GetBinUncertaintyImpl(binidx)`.
   template <bool B = true, class = typename std::enable_if<B && HasBinUncertainty()>::type>
   double GetBinUncertainty(int binidx) const
   {
      return this->GetBinUncertaintyImpl(binidx);
   }
   /// Calculate the bin content's uncertainty for the given bin, using Poisson
   /// statistics on the absolute bin content. Only available if no base provides
   /// this functionality. Requires GetContent().
   template <bool B = true, class = typename std::enable_if<B && !HasBinUncertainty()>::type>
   double GetBinUncertainty(int binidx, ...) const
   {
      auto content = this->GetBinContent(binidx);
      return std::sqrt(std::fabs(content));
   }

   /// Get a view on the statistics values of a bin.
   ConstHistBinStat_t GetView(int idx) const { return ConstHistBinStat_t(*this, idx); }
   /// Get a (non-const) view on the statistics values of a bin.
   HistBinStat_t GetView(int idx) { return HistBinStat_t(*this, idx); }
};
} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
