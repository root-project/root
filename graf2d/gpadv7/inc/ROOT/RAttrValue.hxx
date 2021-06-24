/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrValue
#define ROOT7_RAttrValue

#include <ROOT/RAttrBase.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrValue
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2020-06-24
\brief Template class to access single value from drawable or other attributes
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

template<typename T>
class RAttrValue : public RAttrBase {

   template <typename Q, bool = std::is_enum<Q>::value>
   struct ValueExtractor {
      Q Get(const RAttrMap::Value_t *value)
      {
         return (Q) RAttrMap::Value_t::GetValue<int>(value);
      }
   };

   template <typename Q>
   struct ValueExtractor<Q, false> {
      Q Get(const RAttrMap::Value_t *value) {
         return RAttrMap::Value_t::GetValue<Q>(value);
      }
   };

protected:

   T fDefault{};          ///<!    default value

   RAttrMap CollectDefaults() const override
   {
      return RAttrMap().AddValue(GetName(), fDefault);
   }

   bool IsAggregation() const override { return false; }

public:

   RAttrValue() : RAttrBase(""), fDefault() {}

   RAttrValue(const T& dflt) : RAttrBase(""), fDefault(dflt) {}

   RAttrValue(RDrawable *drawable, const char *name, const T &dflt = T()) : RAttrBase(drawable, name ? name : ""), fDefault(dflt) { }

   RAttrValue(RAttrBase *parent, const char *name, const T &dflt = T()) : RAttrBase(parent, name ? name : ""), fDefault(dflt) { }

   RAttrValue(const RAttrValue& src) : RAttrBase("")
   {
      Set(src.Get());
      fDefault = src.GetDefault();
   }

   T GetDefault() const { return fDefault; }

   void Set(const T &v)
   {
      if (auto access = EnsureAttr(GetName()))
         access.attr->AddValue(access.fullname, v);
   }

   T Get() const
   {
      if (auto v = AccessValue(GetName(), true))
         return ValueExtractor<T>().Get(v.value);
      return fDefault;
   }

   const char *GetName() const { return GetPrefix(); }

   void Clear() override { ClearValue(GetName()); }

   bool Has() const
   {
      if (auto v = AccessValue(GetName(), true)) {
         auto res = RAttrMap::Value_t::GetValue<const RAttrMap::Value_t *,T>(v.value);
         return res ? (res->Kind() != RAttrMap::kNoValue) : false;
      }

      return false;
   }

   RAttrValue &operator=(const T &v) { Set(v); return *this; }

   RAttrValue &operator=(const RAttrValue &v) { Set(v.Get()); return *this; }

   operator T() const { return Get(); }

   friend bool operator==(const RAttrValue& lhs, const RAttrValue& rhs) { return lhs.Get() == rhs.Get(); }
   friend bool operator!=(const RAttrValue& lhs, const RAttrValue& rhs) { return lhs.Get() != rhs.Get(); }

};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RAttrValue
