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

   RAttrMap CollectDefaults() const override;

   bool IsAggregation() const final { return true; }

   void CopyTo(RAttrAggregation &tgt, bool use_style = true) const;

   bool CopyValue(const std::string &name, const RAttrMap::Value_t &value, bool check_type = true);

   bool IsSame(const RAttrAggregation &src, bool use_style = true) const;

   bool IsValueEqual(const std::string &name, const RAttrMap::Value_t &value, bool use_style = false) const;

public:
   RAttrAggregation() = default;

   RAttrAggregation(const RAttrAggregation &src) : RAttrBase() { src.CopyTo(*this); }

   RAttrAggregation(RDrawable *drawable, const char *prefix = nullptr) : RAttrBase(drawable, prefix) {}
   RAttrAggregation(RAttrBase *parent, const char *prefix = nullptr) : RAttrBase(parent, prefix) {}

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

#define R__ATTR_CLASS_DERIVED(ClassName,dflt_prefix,BaseClass) \
protected: \
const RAttrMap &GetDefaults() const override \
{ \
   static auto dflts = CollectDefaults(); \
   return dflts; \
} \
public: \
   ClassName() = default; \
   ClassName(RDrawable *drawable, const char *prefix = dflt_prefix) : BaseClass(drawable, prefix) {} \
   ClassName(RAttrBase *parent, const char *prefix = dflt_prefix) : BaseClass(parent, prefix) {} \
   ClassName(const ClassName &src) : ClassName() { src.CopyTo(*this); } \
   ClassName &operator=(const ClassName &src) { Clear(); src.CopyTo(*this); return *this; }


#define R__ATTR_CLASS(ClassName,dflt_prefix) R__ATTR_CLASS_DERIVED(ClassName,dflt_prefix,RAttrAggregation)

#endif
