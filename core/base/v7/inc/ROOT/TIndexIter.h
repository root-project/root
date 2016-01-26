/// \file ROOT/TIndexIter.h
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2016-01-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TIndexIter
#define ROOT7_TIndexIter

#include <iterator>

namespace ROOT {
namespace Experimental {
namespace Internal {

/**
 \class TIndexIter
 Iterates over an index; the value is defined by the VALUE template parameter.

 Derived classes need to implement `const VALUE& operator*()` and
 `const VALUE* operator->()`. DERIVED must provide a member size_t& GetIndex().
 */

template <class DERIVED, class VALUE>
class TIndexIter:
public std::iterator<std::random_access_iterator_tag, VALUE> {
protected:
  size_t GetIndex() const noexcept { return GetDerived().GetIndex(); }
  DERIVED& GetDerived() noexcept { return *(DERIVED*)this; }
  const DERIVED& GetDerived() const noexcept { return *(DERIVED*)this; }
  static constexpr size_t fgEndIndex = (size_t) -1;

public:
  ///\{
  ///\name Index modifiers
  /// ++i
  TIndexIter &operator++() noexcept {
    ++GetDerived().GetIndex();
    return *this;
  }

  /// --i
  TIndexIter &operator--() noexcept {
    if (GetDerived().GetIndex() != fgEndIndex)
      --GetDerived().GetIndex();
    return *this;
  }

  /// i++
  TIndexIter operator++(int) noexcept {
    TIndexIter old(*this);
    ++(*this);
    return old;
  }

  // i--
  TIndexIter operator--(int) noexcept {
    TIndexIter old(*this);
    --(*this);
    return old;
  }

  TIndexIter &operator+=(int d) noexcept {
    GetDerived().GetIndex() += d;
    return *this;
  }

  TIndexIter &operator-=(int d) noexcept {
    if (d > GetDerived().GetIndex()) {
      GetDerived().GetIndex() = fgEndIndex;
    } else {
      GetDerived().GetIndex() -= d;
    }
    return *this;
  }

  TIndexIter operator+(int d) noexcept {
    TIndexIter ret(*this);
    ret += d;
    return ret;
  }

  TIndexIter operator-(int d) noexcept {
    TIndexIter ret(*this);
    ret -= d;
    return ret;
  }
  ///\}

  friend bool operator<(TIndexIter lhs, TIndexIter rhs) noexcept;
  friend bool operator>(TIndexIter lhs, TIndexIter rhs) noexcept;
  friend bool operator<=(TIndexIter lhs, TIndexIter rhs) noexcept;
  friend bool operator>=(TIndexIter lhs, TIndexIter rhs) noexcept;
  friend bool operator==(TIndexIter lhs, TIndexIter rhs) noexcept;
  friend bool operator!=(TIndexIter lhs, TIndexIter rhs) noexcept;

};

///\{
///\name Relational operators.
template <class DERIVED, class VALUE>
bool operator<(TIndexIter<DERIVED, VALUE> lhs, TIndexIter<DERIVED, VALUE> rhs) noexcept {
  return lhs.GetDerived().GetIndex() < rhs.GetDerived().GetIndex();
}

template <class DERIVED, class VALUE>
bool operator>(TIndexIter<DERIVED, VALUE> lhs, TIndexIter<DERIVED, VALUE> rhs) noexcept {
  return lhs.GetDerived().GetIndex() > rhs.GetDerived().GetIndex();
}

template <class DERIVED, class VALUE>
bool operator<=(TIndexIter<DERIVED, VALUE> lhs, TIndexIter<DERIVED, VALUE> rhs) noexcept {
  return lhs.GetDerived().GetIndex() <= rhs.GetDerived().GetIndex();
}

template <class DERIVED, class VALUE>
inline bool operator>=(TIndexIter<DERIVED, VALUE> lhs, TIndexIter<DERIVED, VALUE> rhs) noexcept {
  return lhs.GetDerived().GetIndex() >= rhs.GetDerived().GetIndex();
}

template <class DERIVED, class VALUE>
inline bool operator==(TIndexIter<DERIVED, VALUE> lhs, TIndexIter<DERIVED, VALUE> rhs) noexcept {
  return lhs.GetDerived().GetIndex() == rhs.GetDerived().GetIndex();
}

template <class DERIVED, class VALUE>
inline bool operator!=(TIndexIter<DERIVED, VALUE> lhs, TIndexIter<DERIVED, VALUE> rhs) noexcept {
  return lhs.GetDerived().GetIndex() != rhs.GetDerived().GetIndex();
}
///\}

}
}
}

#endif // ROOT7_TIndexIter
