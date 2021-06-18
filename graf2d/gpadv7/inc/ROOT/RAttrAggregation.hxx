/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrAggregation
#define ROOT7_RAttrAggregation

#include <ROOT/RAttrBase.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrAggregation
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-18
\brief Base class for attributes aggregations like lines or fill attributes
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RAttrAggregation : public RAttrBase {

protected:
   virtual const RAttrMap &GetDefaults() const;

   virtual RAttrMap CollectDefaults() const;

   void AddDefaultValues(RAttrMap &) const override;

   /// Evaluate attribute value
   template <typename RET_TYPE,typename MATCH_TYPE = void>
   auto Eval(const std::string &name, bool use_dflts = true) const
   {
      if (auto v = AccessValue(name, true))
         return RAttrMap::Value_t::GetValue<RET_TYPE,MATCH_TYPE>(v.value);

      const RAttrMap::Value_t *rec = nullptr;

      if (use_dflts)
         rec = GetDefaults().Find(name);

      return RAttrMap::Value_t::GetValue<RET_TYPE,MATCH_TYPE>(rec);
   }

   template <typename T = void>
   bool HasValue(const std::string &name, bool check_defaults = false) const
   {
      auto res = Eval<const RAttrMap::Value_t *, T>(name, check_defaults);
      return res ? (res->Kind() != RAttrMap::kNoValue) : false;
   }

   template <typename T>
   T GetValue(const std::string &name) const
   {
      return Eval<T>(name);
   }

   void CopyTo(RAttrAggregation &tgt, bool use_style = true) const;

   bool CopyValue(const std::string &name, const RAttrMap::Value_t &value, bool check_type = true);

   bool IsSame(const RAttrAggregation &src, bool use_style = true) const;

   bool IsValueEqual(const std::string &name, const RAttrMap::Value_t &value, bool use_style = false) const;

public:
   RAttrAggregation() = default;

   RAttrAggregation(const RAttrAggregation &src) : RAttrBase() { src.CopyTo(*this); }

   RAttrAggregation &operator=(const RAttrAggregation &src)
   {
      Clear();
      src.CopyTo(*this);
      return *this;
   }

   void Clear() override;

   friend bool operator==(const RAttrAggregation& lhs, const RAttrAggregation& rhs) { return lhs.IsSame(rhs) && rhs.IsSame(lhs); }
   friend bool operator!=(const RAttrAggregation& lhs, const RAttrAggregation& rhs) { return !lhs.IsSame(rhs) || !rhs.IsSame(lhs); }
};

} // namespace Experimental
} // namespace ROOT

#define R__ATTR_CLASS(ClassName,dflt_prefix) \
protected: \
const RAttrMap &GetDefaults() const override \
{ \
   static auto dflts = CollectDefaults(); \
   return dflts; \
} \
public: \
   ClassName() = default; \
   ClassName(RDrawable *drawable, const std::string &prefix = dflt_prefix) { AssignDrawable(drawable, prefix); } \
   ClassName(RAttrBase *parent, const std::string &prefix = dflt_prefix) { AssignParent(parent, prefix); } \
   ClassName(const ClassName &src) : ClassName() { src.CopyTo(*this); } \
   ClassName &operator=(const ClassName &src) { Clear(); src.CopyTo(*this); return *this; } \


#endif
