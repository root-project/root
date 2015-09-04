/// \file ROOT/THistBinIter.h
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-07

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistBinIter
#define ROOT7_THistBinIter

#include <iterator>

namespace ROOT {
namespace Internal {

class THistBinIterBase: public std::iterator<std::random_access_iterator_tag,
   int /*value*/, int /*distance*/, const int* /*pointer*/, const int& /*ref*/> {

protected:
  /*
   TODO: How can I prevent splicing and still make it available?
  ///\{ no splicing
  THistBinIterBase(const THistBinIterBase &) = default;
  THistBinIterBase(THistBinIterBase &&) = default;
  THistBinIterBase &operator=(const THistBinIterBase &) = default;
  THistBinIterBase &operator=(THistBinIterBase &&) = default;
  ///\}
  */

public:
  THistBinIterBase() = default;

  explicit THistBinIterBase(int idx): fCursor(idx) {}

  const int* operator*() const noexcept { return &fCursor; }
  int operator->() const noexcept { return fCursor; }

  friend bool operator<(THistBinIterBase lhs, THistBinIterBase rhs) noexcept;
  friend bool operator>(THistBinIterBase lhs, THistBinIterBase rhs) noexcept;
  friend bool operator<=(THistBinIterBase lhs, THistBinIterBase rhs) noexcept;
  friend bool operator>=(THistBinIterBase lhs, THistBinIterBase rhs) noexcept;
  friend bool operator==(THistBinIterBase lhs, THistBinIterBase rhs) noexcept;
  friend bool operator!=(THistBinIterBase lhs, THistBinIterBase rhs) noexcept;

  int fCursor;
};


bool operator<(THistBinIterBase lhs, THistBinIterBase rhs) noexcept {
  return lhs.fCursor < rhs.fCursor;
}

bool operator>(THistBinIterBase lhs, THistBinIterBase rhs) noexcept {
  return lhs.fCursor > rhs.fCursor;
}

bool operator<=(THistBinIterBase lhs, THistBinIterBase rhs) noexcept {
  return lhs.fCursor <= rhs.fCursor;
}

inline bool operator>=(THistBinIterBase lhs, THistBinIterBase rhs) noexcept {
  return lhs.fCursor >= rhs.fCursor;
}

inline bool operator==(THistBinIterBase lhs, THistBinIterBase rhs) noexcept {
  return lhs.fCursor == rhs.fCursor;
}

inline bool operator!=(THistBinIterBase lhs, THistBinIterBase rhs) noexcept {
  return lhs.fCursor != rhs.fCursor;
}


/// A predicate for THistBinIterBase to accept all bins.
struct HistIterFullRange_t {
  bool operator()(int) const { return true; }
};

/// A bin iterator taking a predicate whether it should skip a bin.
template <class OUTOFRANGE>
class THistBinIter: public THistBinIterBase,
                    private OUTOFRANGE {
  using OutOfRange_t = OUTOFRANGE;
  const OutOfRange_t& GetOutOfRange() const { return *this; }
public:
  THistBinIter() = default;
  THistBinIter(const THistBinIter&) = default;
  THistBinIter(THistBinIter&&) = default;
  explicit THistBinIter(int idx, const OUTOFRANGE& oor = OUTOFRANGE()):
     THistBinIterBase(idx), OutOfRange_t(oor) {}

  /// ++i
  THistBinIter &operator++() noexcept {
    // Could check whether fCursor < fEnd - but what for?
    do {
      ++fCursor;
    } while (GetOutOfRange()(fCursor));
    return *this;
  }

  /// --i
  THistBinIter &operator--() noexcept {
    // Could check whether fCursor > fBegin - but what for?
    do {
      --fCursor;
    } while (GetOutOfRange()(fCursor));
    return *this;
  }

  /// i++
  THistBinIter operator++(int) noexcept {
    THistBinIter old(*this);
    ++(*this);
    return old;
  }

  // i--
  THistBinIter operator--(int) noexcept {
    THistBinIter old(*this);
    --(*this);
    return old;
  }

  THistBinIter &operator+=(int d) noexcept {
    do {
      ++(*this);
      --d;
    } while (d > 0);
    return *this;
  }

  THistBinIter &operator-=(int d) noexcept {
    do {
      --(*this);
      --d;
    } while (d > 0);
    return *this;
  }

  THistBinIter operator+(int d) noexcept {
    THistBinIter ret(*this);
    ret += d;
    return ret;
  }

  THistBinIter operator-(int d) noexcept {
    THistBinIter ret(*this);
    ret -= d;
    return ret;
  }
};
} // namespace Internal
} // namespace ROOT

#endif
