/// \file ROOT/RAxis.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-03-23
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAxis
#define ROOT7_RAxis

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include "ROOT/RStringView.hxx"
#include "ROOT/RLogger.hxx"

namespace ROOT {
namespace Experimental {

/**
 \class RAxisBase
 Histogram axis base class. Keeps track of the number of bins and overflow
 handling. Offers bin iteration.

 Bin indices are starting from 0 for the underflow bin (representing values that
 are lower than the axis range). Starting at index 1 are the actual bins of the
 axis, up to N + 1 for an axis with N bins. Index N + 2 is the overflow bin for
 values larger than the axis range.
  */
class RAxisBase {
public:
   /// Status of FindBin(x)
   enum class EFindStatus {
      kCanGrow, ///< Coordinate could fit after growing the axis
      kValid    ///< The returned bin index is valid
   };

protected:
   ///\name Inaccessible copy, assignment
   /// The copy and move constructors and assignment operators are protected to
   /// prevent slicing.
   ///\{
   RAxisBase(const RAxisBase &) = default;
   RAxisBase(RAxisBase &&) = default;
   RAxisBase &operator=(const RAxisBase &) = default;
   RAxisBase &operator=(RAxisBase &&) = default;
   ///\}

   /// Default construct a RAxisBase (for use by derived classes for I/O)
   RAxisBase() = default;

   /// Virtual destructor needed in this inheritance-based design
   virtual ~RAxisBase() = default;

   /// Construct a RAxisBase.
   ///
   ///\param[in] title - axis title used for graphics and text representation.
   ///\param[in] nbins - number of bins in this axis, including under- and
   /// overflow bins.
   RAxisBase(std::string_view title, int nbins) noexcept
      : fNBins(nbins), fTitle(title)
   {}

   /// Construct a RAxisBase.
   ///
   ///\param[in] nbins - number of bins in this axis, including under- and
   /// overflow bins.
   RAxisBase(int nbins) noexcept: RAxisBase("", nbins) {}

   /// Given rawbin (<0 for underflow, >= GetNBinsNoOver() for overflow), determine the
   /// actual bin number taking into account how over/underflow should be
   /// handled.
   ///
   /// \param[out] status result status of the bin determination.
   /// \return Returns the bin number adjusted for potential over- and underflow
   /// bins. Returns kIgnoreBin if the axis cannot handle the over- / underflow,
   /// in which case `status` will tell how to deal with this overflow.
   int AdjustOverflowBinNumber(double rawbin) const
   {
      if (rawbin < 0)
         return 0;
      // Take underflow into account.
      ++rawbin;

      if (rawbin >= GetNBins())
         return GetNBins() - 1;

      return (int)rawbin;
   }

public:
   /**
    \class const_iterator
    Random const_iterator through bins. Represents the bin index, not a bin
    content: the axis has no notion of any content.
    */
   class const_iterator: public std::iterator<std::random_access_iterator_tag, int /*value*/, int /*distance*/,
                                              const int * /*pointer*/, const int & /*ref*/> {
      int fCursor = 0; ///< Current iteration position

   public:
      const_iterator() = default;

      /// Initialize a const_iterator with its position
      explicit const_iterator(int cursor) noexcept: fCursor(cursor) {}

      /// ++i
      const_iterator &operator++() noexcept
      {
         // Could check whether fCursor < fEnd - but what for?
         ++fCursor;
         return *this;
      }

      /// --i
      const_iterator &operator--() noexcept
      {
         // Could check whether fCursor > fBegin - but what for?
         --fCursor;
         return *this;
      }

      /// i++
      const_iterator operator++(int)noexcept
      {
         const_iterator old(*this);
         ++(*this);
         return old;
      }

      // i--
      const_iterator operator--(int)noexcept
      {
         const_iterator old(*this);
         --(*this);
         return old;
      }

      // i += 2
      const_iterator &operator+=(int d) noexcept
      {
         fCursor += d;
         return *this;
      }

      // i -= 2
      const_iterator &operator-=(int d) noexcept
      {
         fCursor -= d;
         return *this;
      }

      // i + 2
      const_iterator operator+(int d) noexcept
      {
         const_iterator ret(*this);
         ret += d;
         return ret;
      }

      // i - 2
      const_iterator operator-(int d) noexcept
      {
         const_iterator ret(*this);
         ret -= d;
         return ret;
      }

      // *i
      const int *operator*() const noexcept { return &fCursor; }

      // i->
      int operator->() const noexcept { return fCursor; }

      friend bool operator<(const_iterator lhs, const_iterator rhs) noexcept;
      friend bool operator>(const_iterator lhs, const_iterator rhs) noexcept;
      friend bool operator<=(const_iterator lhs, const_iterator rhs) noexcept;
      friend bool operator>=(const_iterator lhs, const_iterator rhs) noexcept;
      friend bool operator==(const_iterator lhs, const_iterator rhs) noexcept;
      friend bool operator!=(const_iterator lhs, const_iterator rhs) noexcept;
   };

   /// FindBin() returns this bin to signal that the bin number is invalid.
   constexpr static const int kIgnoreBin = -1;

   /// Extra bins for each EAxisOverflow value.
   constexpr static const int kNOverflowBins[4] = {0, 1, 1, 2};

   /// Get the axis's title
   const std::string &GetTitle() const { return fTitle; }

   /// Whether this axis can grow (and thus has no overflow bins).
   virtual bool CanGrow() const noexcept = 0;

   /// Get the number of bins, excluding under- and overflow.
   int GetNBinsNoOver() const noexcept { return fNBins - GetNOverflowBins(); }

   /// Get the number of bins, including under- and overflow.
   int GetNBins() const noexcept { return fNBins; }

   /// Get the number of over- and underflow bins: 0 for growable axes, 2 otherwise.
   int GetNOverflowBins() const noexcept
   {
      if (CanGrow())
         return 0;
      else
         return 2;
   };

   /// Get the bin index for the underflow bin.
   int GetUnderflowBin() const noexcept { return 0; }

   /// Get the bin index for the underflow bin (or the next bin outside range
   /// if CanGrow()).
   int GetOverflowBin() const noexcept { return GetNBinsNoOver() + 1; }

   /// Whether the bin index is referencing a bin lower than the axis range.
   bool IsUnderflowBin(int bin) const noexcept { return bin <= GetUnderflowBin(); }

   /// Whether the bin index is referencing a bin higher than the axis range.
   bool IsOverflowBin(int bin) const noexcept { return bin >= GetOverflowBin(); }

   ///\name Iterator interfaces
   ///\{

   /// Get a const_iterator pointing to the first non-underflow bin.
   const_iterator begin() const noexcept { return const_iterator{1}; }

   /// Get a const_iterator pointing the underflow bin.
   const_iterator begin_with_underflow() const noexcept { return const_iterator{0}; }

   /// Get a const_iterator pointing right beyond the last non-overflow bin
   /// (i.e. pointing to the overflow bin).
   const_iterator end() const noexcept { return const_iterator{GetOverflowBin()}; }

   /// Get a const_iterator pointing right beyond the overflow bin.
   const_iterator end_with_overflow() const noexcept { return const_iterator{GetOverflowBin() + 1}; }
   ///\}

private:
   unsigned int fNBins;   ///< Number of bins including under- and overflow.
   std::string fTitle;    ///< Title of this axis, used for graphics / text.
};

///\name RAxisBase::const_iterator comparison operators
///\{

/// i < j
inline bool operator<(RAxisBase::const_iterator lhs, RAxisBase::const_iterator rhs) noexcept
{
   return lhs.fCursor < rhs.fCursor;
}

/// i > j
inline bool operator>(RAxisBase::const_iterator lhs, RAxisBase::const_iterator rhs) noexcept
{
   return lhs.fCursor > rhs.fCursor;
}

/// i <= j
inline bool operator<=(RAxisBase::const_iterator lhs, RAxisBase::const_iterator rhs) noexcept
{
   return lhs.fCursor <= rhs.fCursor;
}

/// i >= j
inline bool operator>=(RAxisBase::const_iterator lhs, RAxisBase::const_iterator rhs) noexcept
{
   return lhs.fCursor >= rhs.fCursor;
}

/// i == j
inline bool operator==(RAxisBase::const_iterator lhs, RAxisBase::const_iterator rhs) noexcept
{
   return lhs.fCursor == rhs.fCursor;
}

/// i != j
inline bool operator!=(RAxisBase::const_iterator lhs, RAxisBase::const_iterator rhs) noexcept
{
   return lhs.fCursor != rhs.fCursor;
}
///\}

/**
\class RAxisConfig
Objects used to configure the different axis types. It can store the
properties of all possible axis types, together with the type of the axis.

RODO: that's what a variant will be invented for!
*/
class RAxisConfig: public RAxisBase {
public:
   enum EKind {
      kEquidistant, ///< represents a RAxisEquidistant
      kGrow,        ///< represents a RAxisGrow
      kIrregular,   ///< represents a RAxisIrregular
      kLabels,      ///< represents a RAxisLabels
      kNumKinds
   };

private:
   EKind fKind;                      ///< The kind of axis represented by this configuration
   std::vector<double> fBinBorders;  ///< Bin borders of the RAxisIrregular
   std::vector<std::string> fLabels; ///< Bin labels for a RAxisLabels

   /// Represents a `RAxisEquidistant` with `nbins` from `from` to `to`, and
   /// axis title.
   explicit RAxisConfig(std::string_view title, int nbins, double from, double to, EKind kind)
      : RAxisBase(title, nbins + ((kind == kGrow) ? 0 : 2)), fKind(kind), fBinBorders(2)
   {
      if (from > to)
         std::swap(to, from);

      fBinBorders[0] = from;
      fBinBorders[1] = to;
   }

public:
   /// Tag type signalling that an axis should be able to grow; used for calling
   /// the appropriate constructor.
   struct Grow_t {
   };
   /// Tag signalling that an axis should be able to grow; used for calling the
   /// appropriate constructor like so:
   ///     RAxisConfig ac(RAxisConfig::Grow, 10, 0., 1.);
   constexpr static const Grow_t Grow{};

   /// Represents a `RAxisEquidistant` with `nbins` from `from` to `to`, and
   /// axis title.
   RAxisConfig(std::string_view title, int nbins, double from, double to)
      : RAxisConfig(title, nbins, from, to, kEquidistant)
   {}

   /// Represents a `RAxisEquidistant` with `nbins` from `from` to `to`.
   RAxisConfig(int nbins, double from, double to): RAxisConfig("", nbins, from, to, kEquidistant) {}

   /// Represents a `RAxisGrow` with `nbins` from `from` to `to`, and axis title.
   RAxisConfig(std::string_view title, Grow_t, int nbins, double from, double to)
      : RAxisConfig(title, nbins, from, to, kGrow)
   {}

   /// Represents a `RAxisGrow` with `nbins` from `from` to `to`.
   RAxisConfig(Grow_t, int nbins, double from, double to): RAxisConfig("", nbins, from, to, kGrow) {}

   /// Represents a `RAxisIrregular` with `binborders` and title.
   RAxisConfig(std::string_view title, const std::vector<double> &binborders)
      : RAxisBase(title, binborders.size() + 1), fKind(kIrregular), fBinBorders(binborders)
   {}

   /// Represents a `RAxisIrregular` with `binborders`.
   RAxisConfig(const std::vector<double> &binborders): RAxisConfig("", binborders) {}

   /// Represents a `RAxisIrregular` with `binborders` and title.
   RAxisConfig(std::string_view title, std::vector<double> &&binborders) noexcept
      : RAxisBase(title, binborders.size() + 1), fKind(kIrregular),
        fBinBorders(std::move(binborders))
   {}

   /// Represents a `RAxisIrregular` with `binborders`.
   RAxisConfig(std::vector<double> &&binborders) noexcept: RAxisConfig("", std::move(binborders)) {}

   /// Represents a `RAxisLabels` with `labels` and title.
   RAxisConfig(std::string_view title, const std::vector<std::string_view> &labels)
      : RAxisBase(title, labels.size()), fKind(kLabels), fLabels(labels.begin(), labels.end())
   {}

   /// Represents a `RAxisLabels` with `labels`.
   RAxisConfig(const std::vector<std::string_view> &labels): RAxisConfig("", labels) {}

   /// Represents a `RAxisLabels` with `labels` and title.
   RAxisConfig(std::string_view title, std::vector<std::string> &&labels)
      : RAxisBase(title, labels.size()), fKind(kLabels), fLabels(std::move(labels))
   {}

   /// Represents a `RAxisLabels` with `labels`.
   RAxisConfig(std::vector<std::string> &&labels): RAxisConfig("", std::move(labels)) {}

   /// Whether this axis can grow depends on which constructor was called
   bool CanGrow() const noexcept final override {
      switch(fKind) {
         case kEquidistant:
         case kIrregular:
            return false;

         case kGrow:
         case kLabels:
            return true;

         case kNumKinds:
            R__ERROR_HERE("HIST") << "Impossible axis kind!";
      }
      return false;
   }

   /// Get the axis kind represented by this `RAxisConfig`.
   EKind GetKind() const noexcept { return fKind; }

   /// Get the bin borders; non-empty if the GetKind() == kIrregular.
   const std::vector<double> &GetBinBorders() const noexcept { return fBinBorders; }

   /// Get the bin labels; non-empty if the GetKind() == kLabels.
   const std::vector<std::string> &GetBinLabels() const noexcept { return fLabels; }
};

/**
 Axis with equidistant bin borders. Defined by lower l and upper u limit and
 the number of bins n. All bins have the same width (u-l)/n.

 This axis cannot grow; use `RAxisGrow` for that.
 */
class RAxisEquidistant: public RAxisBase {
protected:
   double fLow = 0.;         ///< The lower limit of the axis
   double fInvBinWidth = 0.; ///< The inverse of the bin width

   /// Determine the inverse bin width.
   /// \param nbinsNoOver - number of bins without unter-/overflow
   /// \param lowOrHigh - first axis boundary
   /// \param lighOrLow - second axis boundary
   static double GetInvBinWidth(int nbinsNoOver, double lowOrHigh, double highOrLow)
   {
      return nbinsNoOver / std::abs(highOrLow - lowOrHigh);
   }

   /// Initialize a RAxisEquidistant.
   /// \param[in] title - axis title used for graphics and text representation.
   /// \param nbins - number of bins in the axis, excluding under- and overflow
   ///   bins.
   /// \param low - the low axis range. Any coordinate below that is considered
   ///   as underflow. The first bin's lower edge is at this value.
   /// \param high - the high axis range. Any coordinate above that is considered
   ///   as overflow. The last bin's higher edge is at this value.
   explicit RAxisEquidistant(std::string_view title, int nbinsNoOver, double low, double high, bool canGrow) noexcept
      : RAxisBase(title, nbinsNoOver + (canGrow ? 0 : 2)), fLow(low), fInvBinWidth(GetInvBinWidth(nbinsNoOver, low, high))
   {}

   /// Initialize a RAxisEquidistant.
   /// \param nbins - number of bins in the axis, excluding under- and overflow
   ///   bins.
   /// \param low - the low axis range. Any coordinate below that is considered
   ///   as underflow. The first bin's lower edge is at this value.
   /// \param high - the high axis range. Any coordinate above that is considered
   ///   as overflow. The last bin's higher edge is at this value.
   explicit RAxisEquidistant(int nbinsNoOver, double low, double high, bool canGrow) noexcept
      : RAxisEquidistant("", nbinsNoOver, low, high, canGrow)
   {}

public:
   RAxisEquidistant() = default;

   /// Initialize a RAxisEquidistant.
   /// \param nbins - number of bins in the axis, excluding under- and overflow
   ///   bins.
   /// \param low - the low axis range. Any coordinate below that is considered
   ///   as underflow. The first bin's lower edge is at this value.
   /// \param high - the high axis range. Any coordinate above that is considered
   ///   as overflow. The last bin's higher edge is at this value.
   /// \param canGrow - whether this axis can extend its range.
   explicit RAxisEquidistant(int nbinsNoOver, double low, double high) noexcept
      : RAxisEquidistant(nbinsNoOver, low, high, false /*canGrow*/)
   {}

   /// Initialize a RAxisEquidistant.
   /// \param[in] title - axis title used for graphics and text representation.
   /// \param nbins - number of bins in the axis, excluding under- and overflow
   ///   bins.
   /// \param low - the low axis range. Any coordinate below that is considered
   ///   as underflow. The first bin's lower edge is at this value.
   /// \param high - the high axis range. Any coordinate above that is considered
   ///   as overflow. The last bin's higher edge is at this value.
   explicit RAxisEquidistant(std::string_view title, int nbinsNoOver, double low, double high) noexcept
      : RAxisEquidistant(title, nbinsNoOver, low, high, false /*canGrow*/)
   {}

   /// Convert to RAxisConfig.
   operator RAxisConfig() const { return RAxisConfig(GetNBinsNoOver(), GetMinimum(), GetMaximum()); }

   /// Find the bin index for the given coordinate.
   /// \note Passing a bin border coordinate can either return the bin above or
   /// below the bin border. I.e. don't do that for reliable results!
   int FindBin(double x) const noexcept
   {
      double rawbin = (x - fLow) * fInvBinWidth;
      return AdjustOverflowBinNumber(rawbin);
   }

   /// This axis cannot grow.
   bool CanGrow() const noexcept override { return false; }

   /// Get the low end of the axis range.
   double GetMinimum() const noexcept { return fLow; }

   /// Get the high end of the axis range.
   double GetMaximum() const noexcept { return fLow + GetNBinsNoOver() / fInvBinWidth; }

   /// Get the width of the bins
   double GetBinWidth() const noexcept { return 1. / fInvBinWidth; }

   /// Get the inverse of the width of the bins
   double GetInverseBinWidth() const noexcept { return fInvBinWidth; }

   /// Get the bin center for the given bin index.
   /// For the bin == 1 (the first bin) of 2 bins for an axis (0., 1.), this
   /// returns 0.25.
   double GetBinCenter(int bin) const noexcept { return fLow + (bin - 0.5) / fInvBinWidth; }

   /// Get the low bin border for the given bin index.
   /// For the bin == 1 (the first bin) of 2 bins for an axis (0., 1.), this
   /// returns 0.
   double GetBinFrom(int bin) const noexcept { return fLow + (bin - 1) / fInvBinWidth; }

   /// Get the high bin border for the given bin index.
   /// For the bin == 1 (the first bin) of 2 bins for an axis (0., 1.), this
   /// returns 0.5.
   double GetBinTo(int bin) const noexcept { return GetBinFrom(bin + 1); }

   int GetBinIndexForLowEdge(double x) const noexcept;
};

/// Equality-compare two RAxisEquidistant.
inline bool operator==(const RAxisEquidistant &lhs, const RAxisEquidistant &rhs) noexcept
{
   return lhs.GetNBins() == rhs.GetNBins() && lhs.GetMinimum() == rhs.GetMinimum() &&
          lhs.GetInverseBinWidth() == rhs.GetInverseBinWidth();
}

/** An axis that can extend its range, keeping the number of its bins unchanged.
 The axis is constructed with an initial range. Apart from its ability to
 grow, this axis behaves like a RAxisEquidistant.
 */
class RAxisGrow: public RAxisEquidistant {
public:
   /// Initialize a RAxisGrow.
   /// \param nbins - number of bins in the axis, excluding under- and overflow
   ///   bins. This value is fixed over the lifetime of the object.
   /// \param low - the initial value for the low axis range. Any coordinate
   ///   below that is considered as underflow. To trigger the growing of the
   ///   axis call Grow().
   /// \param high - the initial value for the high axis range. Any coordinate
   ///   above that is considered as overflow. To trigger the growing of the
   ///   axis call Grow()
   explicit RAxisGrow(std::string_view title, int nbins, double low, double high) noexcept
      : RAxisEquidistant(title, nbins, low, high, CanGrow())
   {}

   /// Initialize a RAxisGrow.
   /// \param[in] title - axis title used for graphics and text representation.
   /// \param nbins - number of bins in the axis, excluding under- and overflow
   ///   bins. This value is fixed over the lifetime of the object.
   /// \param low - the initial value for the low axis range. Any coordinate
   ///   below that is considered as underflow. To trigger the growing of the
   ///   axis call Grow().
   /// \param high - the initial value for the high axis range. Any coordinate
   ///   above that is considered as overflow. To trigger the growing of the
   ///   axis call Grow()
   explicit RAxisGrow(int nbins, double low, double high) noexcept: RAxisEquidistant(nbins, low, high, CanGrow()) {}

   /// Convert to RAxisConfig.
   operator RAxisConfig() const { return RAxisConfig(RAxisConfig::Grow, GetNBinsNoOver(), GetMinimum(), GetMaximum()); }

   /// Grow this axis to make the "virtual bin" toBin in-range. This keeps the
   /// non-affected axis limit unchanged, and extends the other axis limit such
   /// that a number of consecutive bins are merged.
   ///
   /// Example, assuming an initial RAxisGrow with 10 bins from 0. to 1.:
   ///   - `Grow(0)`: that (virtual) bin spans from -0.1 to 0. To include it
   ///     in the axis range, the lower limit must be shifted. The minimal number
   ///     of bins that can be merged is 2, thus the new axis will span from
   ///     -1. to 1.
   ///   - `Grow(-1)`: that (virtual) bin spans from -0.2 to 0.1. To include it
   ///     in the axis range, the lower limit must be shifted. The minimal number
   ///     of bins that can be merged is 2, thus the new axis will span from
   ///     -1. to 1.
   ///   - `Grow(50)`: that (virtual) bin spans from 4.9 to 5.0. To include it
   ///     in the axis range, the higher limit must be shifted. Five bins need to
   ///     be merged, making the new axis range 0. to 5.0.
   ///
   /// \param toBin - the "virtual" bin number, as if the axis had an infinite
   ///   number of bins with the current bin width. For instance, for an axis
   ///   with ten bins in the range 0. to 1., the coordinate 2.05 has the virtual
   ///   bin index 20.
   /// \return Returns the number of bins that were merged to reach the value.
   ///   A value of 1 means that no bins were merged (toBin was in the original
   ///   axis range).
   int Grow(int toBin);

   /// This axis kind can increase its range.
   bool CanGrow() const noexcept final override { return true; }
};

/**
  An axis with non-equidistant bins (also known as "variable binning"). It is
  defined by an array of bin borders - one more than the number of
  (non-overflow-) bins it has! As an example, an axis with two bin needs three
  bin borders:
    - lower edge of the first bin;
    - higher edge of the first bin, identical to the lower edge of the second
      bin;
    - higher edge of the second bin

  This axis cannot grow; the size of new bins would not be well defined.
 */
class RAxisIrregular: public RAxisBase {
private:
   /// Bin borders, one more than the number of non-overflow bins.
   std::vector<double> fBinBorders;

public:
   RAxisIrregular() = default;

   /// Construct a RAxisIrregular from a vector of bin borders.
   /// \note The bin borders must be sorted in increasing order!
   explicit RAxisIrregular(const std::vector<double> &binborders)
      : RAxisBase(binborders.size() + 1), fBinBorders(binborders)
   {
#ifdef R__DO_RANGE_CHECKS
      if (!std::is_sorted(fBinBorders.begin(), fBinBorders.end()))
         R__ERROR_HERE("HIST") << "Bin borders must be sorted!";
#endif // R__DO_RANGE_CHECKS
   }

   /// Construct a RAxisIrregular from a vector of bin borders.
   /// \note The bin borders must be sorted in increasing order!
   /// Faster, noexcept version taking an rvalue of binborders. The compiler will
   /// know when it can take this one.
   explicit RAxisIrregular(std::vector<double> &&binborders) noexcept
      : RAxisBase(binborders.size() + 1), fBinBorders(std::move(binborders))
   {
#ifdef R__DO_RANGE_CHECKS
      if (!std::is_sorted(fBinBorders.begin(), fBinBorders.end()))
         R__ERROR_HERE("HIST") << "Bin borders must be sorted!";
#endif // R__DO_RANGE_CHECKS
   }

   /// Construct a RAxisIrregular from a vector of bin borders.
   /// \note The bin borders must be sorted in increasing order!
   explicit RAxisIrregular(std::string_view title, const std::vector<double> &binborders)
      : RAxisBase(title, binborders.size() + 1), fBinBorders(binborders)
   {
#ifdef R__DO_RANGE_CHECKS
      if (!std::is_sorted(fBinBorders.begin(), fBinBorders.end()))
         R__ERROR_HERE("HIST") << "Bin borders must be sorted!";
#endif // R__DO_RANGE_CHECKS
   }

   /// Construct a RAxisIrregular from a vector of bin borders.
   /// \note The bin borders must be sorted in increasing order!
   /// Faster, noexcept version taking an rvalue of binborders. The compiler will
   /// know when it can take this one.
   explicit RAxisIrregular(std::string_view title, std::vector<double> &&binborders) noexcept
      : RAxisBase(title, binborders.size() + 1), fBinBorders(std::move(binborders))
   {
#ifdef R__DO_RANGE_CHECKS
      if (!std::is_sorted(fBinBorders.begin(), fBinBorders.end()))
         R__ERROR_HERE("HIST") << "Bin borders must be sorted!";
#endif // R__DO_RANGE_CHECKS
   }

   /// Convert to RAxisConfig.
   operator RAxisConfig() const { return RAxisConfig(GetBinBorders()); }

   /// Find the bin index corresponding to coordinate x. If the coordinate is
   /// below the axis range, return 0. If it is above, return N + 1 for an axis
   /// with N non-overflow bins.
   int FindBin(double x) const noexcept
   {
      const auto bBegin = fBinBorders.begin();
      const auto bEnd = fBinBorders.end();
      // lower_bound finds the first bin border that is >= x.
      auto iNotLess = std::lower_bound(bBegin, bEnd, x);
      int rawbin = iNotLess - bBegin;
      // No need for AdjustOverflowBinNumber(rawbin) here; lower_bound() is the
      // answer: e.g. for x < *bBegin, rawbin is 0.
      return rawbin;
   }

   /// Get the bin center of the bin with the given index.
   ///
   /// For the bin at index 0 (i.e. the underflow bin), a bin center of
   /// `std::numeric_limits<double>::min()` is returned, i.e. the minimum value
   /// that can be held in a double.
   /// Similarly, for the bin at index N + 1 (i.e. the overflow bin), a bin
   /// center of `std::numeric_limits<double>::max()` is returned, i.e. the
   /// maximum value that can be held in a double.
   double GetBinCenter(int bin) const noexcept
   {
      if (IsUnderflowBin(bin))
         return std::numeric_limits<double>::min();
      if (IsOverflowBin(bin))
         return std::numeric_limits<double>::max();
      return 0.5 * (fBinBorders[bin - 1] + fBinBorders[bin]);
   }

   /// Get the lower bin border for a given bin index.
   ///
   /// For the bin at index 0 (i.e. the underflow bin), a lower bin border of
   /// `std::numeric_limits<double>::min()` is returned, i.e. the minimum value
   /// that can be held in a double.
   double GetBinFrom(int bin) const noexcept
   {
      if (IsUnderflowBin(bin))
         return std::numeric_limits<double>::min();
      // bin 0 is underflow;
      // bin 1 starts at fBinBorders[0]
      return fBinBorders[bin - 1];
   }

   /// Get the higher bin border for a given bin index.
   ///
   /// For the bin at index N + 1 (i.e. the overflow bin), a bin border of
   /// `std::numeric_limits<double>::max()` is returned, i.e. the maximum value
   /// that can be held in a double.
   double GetBinTo(int bin) const noexcept
   {
      if (IsOverflowBin(bin))
         return std::numeric_limits<double>::max();
      return GetBinFrom(bin + 1);
   }

   /// This axis cannot be extended.
   bool CanGrow() const noexcept final override { return false; }

   /// Access to the bin borders used by this axis.
   const std::vector<double> &GetBinBorders() const noexcept { return fBinBorders; }
};

/**
 \class RAxisLabels
 A RAxisGrow that has a label assigned to each bin and a bin width of 1.

 While filling still works through coordinates (i.e. arrays of doubles),
 RAxisLabels allows to convert a string to a bin number or the bin's coordinate
 center. The number of labels and the number of bins reported by RAxisGrow might
 differ: the RAxisGrow will only grow when seeing a Fill(), while the RAxisLabels
 will add a new label whenever `GetBinCenter()` is called.

 Implementation details:
 Filling happens often; GetBinCenter() needs to be fast. Thus the unordered_map.
 The painter needs the reverse: it wants the label for bin 0, bin 1 etc. The axis
 should only store the bin labels once; referencing them is (due to re-allocation,
 hashing etc) non-trivial. So instead, build a vector<string_view> for the few
 times the axis needs to be painted.
 */
class RAxisLabels: public RAxisGrow {
private:
   /// Map of label (view on `fLabels`'s elements) to bin index
   std::unordered_map<std::string, int /*bin number*/> fLabelsIndex;

public:
   /// Construct a RAxisLables from a `vector` of `string_view`s, with title.
   explicit RAxisLabels(std::string_view title, const std::vector<std::string_view> &labels)
      : RAxisGrow(title, labels.size(), 0., static_cast<double>(labels.size()))
   {
      for (size_t i = 0, n = labels.size(); i < n; ++i)
         fLabelsIndex[std::string(labels[i])] = i;
   }

   /// Construct a RAxisLables from a `vector` of `string`s, with title.
   explicit RAxisLabels(std::string_view title, const std::vector<std::string> &labels)
      : RAxisGrow(title, labels.size(), 0., static_cast<double>(labels.size()))
   {
      for (size_t i = 0, n = labels.size(); i < n; ++i)
         fLabelsIndex[labels[i]] = i;
   }

   /// Construct a RAxisLables from a `vector` of `string_view`s
   explicit RAxisLabels(const std::vector<std::string_view> &labels): RAxisLabels("", labels) {}

   /// Construct a RAxisLables from a `vector` of `string`s
   explicit RAxisLabels(const std::vector<std::string> &labels): RAxisLabels("", labels) {}

   /// Get the bin index with label.
   int GetBinIndex(const std::string &label)
   {
      auto insertResult = fLabelsIndex.insert({label, -1});
      if (insertResult.second) {
         // we have created a new label
         int idx = fLabelsIndex.size() - 1;
         insertResult.first->second = idx;
         return idx;
      }
      return insertResult.first->second;
   }

   /// Get the center of the bin with label.
   double GetBinCenter(const std::string &label)
   {
      return GetBinIndex(label) - 0.5; // bin *center*
   }

   /// Build a vector of labels. The position in the vector defines the label's bin.
   std::vector<std::string_view> GetBinLabels() const
   {
      std::vector<std::string_view> vec(fLabelsIndex.size());
      for (const auto &kv: fLabelsIndex)
         vec.at(kv.second) = kv.first;
      return vec;
   }
};

namespace Internal {

/// Converts a RAxisConfig of whatever kind to the corresponding RAxisBase-derived
/// object.
template <RAxisConfig::EKind>
struct AxisConfigToType; // Only specializations are defined.

template <>
struct AxisConfigToType<RAxisConfig::kEquidistant> {
   using Axis_t = RAxisEquidistant;

   Axis_t operator()(const RAxisConfig &cfg) noexcept
   {
      return RAxisEquidistant(cfg.GetTitle(), cfg.GetNBinsNoOver(), cfg.GetBinBorders()[0], cfg.GetBinBorders()[1]);
   }
};

template <>
struct AxisConfigToType<RAxisConfig::kGrow> {
   using Axis_t = RAxisGrow;

   Axis_t operator()(const RAxisConfig &cfg) noexcept
   {
      return RAxisGrow(cfg.GetTitle(), cfg.GetNBinsNoOver(), cfg.GetBinBorders()[0], cfg.GetBinBorders()[1]);
   }
};
template <>
struct AxisConfigToType<RAxisConfig::kIrregular> {
   using Axis_t = RAxisIrregular;

   Axis_t operator()(const RAxisConfig &cfg) { return RAxisIrregular(cfg.GetTitle(), cfg.GetBinBorders()); }
};

template <>
struct AxisConfigToType<RAxisConfig::kLabels> {
   using Axis_t = RAxisLabels;

   Axis_t operator()(const RAxisConfig &cfg) { return RAxisLabels(cfg.GetTitle(), cfg.GetBinLabels()); }
};

} // namespace Internal

/// Common view on a RAxis, no matter what its kind.
class RAxisView {
   /// View on a `RAxisEquidistant`, `RAxisGrow` or `RAxisLabel`.
   const RAxisEquidistant *fEqui = nullptr;
   /// View on a `RAxisIrregular`.
   const RAxisIrregular *fIrr = nullptr;

public:
   RAxisView() = default;

   /// Construct a view on a `RAxisEquidistant`, `RAxisGrow` or `RAxisLabel`.
   RAxisView(const RAxisEquidistant &equi): fEqui(&equi) {}

   /// Construct a view on a `RAxisIrregular`.
   RAxisView(const RAxisIrregular &irr): fIrr(&irr) {}

   const std::string &GetTitle() const { return fEqui ? fEqui->GetTitle() : fIrr->GetTitle(); }

   /// Find the bin containing coordinate `x`. Forwards to the underlying axis.
   int FindBin(double x) const noexcept
   {
      if (fEqui)
         return fEqui->FindBin(x);
      return fIrr->FindBin(x);
   }

   /// Get the number of bins. Forwards to the underlying axis.
   int GetNBins() const noexcept
   {
      if (fEqui)
         return fEqui->GetNBins();
      return fIrr->GetNBins();
   }

   /// Get the lower axis limit.
   double GetFrom() const { return GetBinFrom(1); }
   /// Get the upper axis limit.
   double GetTo() const { return GetBinTo(GetNBins() - 2); }

   /// Get the bin center of bin index `i`. Forwards to the underlying axis.
   double GetBinCenter(int i) const noexcept
   {
      if (fEqui)
         return fEqui->GetBinCenter(i);
      return fIrr->GetBinCenter(i);
   }

   /// Get the minimal coordinate of bin index `i`. Forwards to the underlying axis.
   double GetBinFrom(int i) const noexcept
   {
      if (fEqui)
         return fEqui->GetBinFrom(i);
      return fIrr->GetBinFrom(i);
   }

   /// Get the maximal coordinate of bin index `i`. Forwards to the underlying axis.
   double GetBinTo(int i) const noexcept
   {
      if (fEqui)
         return fEqui->GetBinTo(i);
      return fIrr->GetBinTo(i);
   }

   /// Get the axis as a RAxisEquidistant; returns nullptr if it's a RAxisIrregular.
   const RAxisEquidistant *GetAsEquidistant() const { return fEqui; }
   /// Get the axis as a RAxisIrregular; returns nullptr if it's a RAxisEquidistant.
   const RAxisIrregular *GetAsIrregular() const { return fIrr; }
};

///\name Axis Compatibility
///\{
enum class EAxisCompatibility {
   kIdentical, ///< Source and target axes are identical

   kContains, ///< The source is a subset of bins of the target axis

   /// The bins of the source axis have finer granularity, but the bin borders
   /// are compatible. Example:
   /// source: 0., 1., 2., 3., 4., 5., 6.; target: 0., 2., 5., 6.
   /// Note that this is *not* a symmetrical property: only one of
   /// CanMerge(source, target), CanMap(target, source) can return kContains.
   kSampling,

   /// The source axis and target axis have different binning. Example:
   /// source: 0., 1., 2., 3., 4., target: 0., 0.1, 0.2, 0.3, 0.4
   kIncompatible
};

EAxisCompatibility CanMap(RAxisEquidistant &target, RAxisEquidistant &source) noexcept;
///\}

} // namespace Experimental
} // namespace ROOT

#endif
