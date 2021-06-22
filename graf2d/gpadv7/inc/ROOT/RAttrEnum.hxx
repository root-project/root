/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrEnum
#define ROOT7_RAttrEnum

#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrValue
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-21
\brief Template class to access single enum value from drawable or other attributes aggregations
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

template<typename T, class = typename std::enable_if<std::is_enum<T>::value>::type>
class RAttrEnum : public RAttrValue<T> {

protected:

   T fMaximum;                ///<!  maximal value

public:

   RAttrEnum() : RAttrValue<T>(), fMaximum() { }

   RAttrEnum(T dflt, T max) : RAttrValue<T>(dflt), fMaximum(max) { }

   RAttrEnum(RDrawable *drawable, const char *name, T dflt, T max) : RAttrValue<T>(drawable, name, dflt), fMaximum(max) { }

   RAttrEnum(RAttrBase *parent, const char *name,  T dflt, T max) : RAttrValue<T>(parent, name, dflt), fMaximum(max) { }

   RAttrEnum(const RAttrEnum& src) = default;

   T GetMaximum() const { return fMaximum; }

   T Get() const
   {
      auto v = RAttrValue<T>::Get();
      return (v >= T()) && (v <= fMaximum) ? v : RAttrValue<T>::GetDefault();
   }

   RAttrEnum &operator=(const T &v) { RAttrValue<T>::Set(v); return *this; }

   RAttrEnum &operator=(const RAttrEnum &v) { RAttrValue<T>::Set(v.Get()); return *this; }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RAttrEnum
