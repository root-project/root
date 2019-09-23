/// \file ROOT/RAttrBase.hxx
/// \ingroup Gpad ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2019-09-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrBase
#define ROOT7_RAttrBase

#include <ROOT/RAttrMap.hxx>
#include <ROOT/RStyle.hxx>
#include <ROOT/RDrawable.hxx>

namespace ROOT {
namespace Experimental {

/** Base class for all attributes, used with RDrawable */
class RAttrBase {

   friend class RAttrMap;

   RDrawable *fDrawable{nullptr};      ///<! drawable used to store attributes
   std::unique_ptr<RAttrMap> fOwnAttr; ///<! own instance when deep copy is created
   std::string fPrefix;                ///<! name prefix for all attributes values
   const RAttrBase *fParent{nullptr};  ///<! parent attributes, prefix applied to it

protected:

   virtual const RAttrMap &GetDefaults() const;

   bool CopyValue(const std::string &name, const RAttrMap::Value_t &value, bool check_type = true);

   bool IsValueEqual(const std::string &name, const RAttrMap::Value_t *value, bool use_style = false) const;

   ///////////////////////////////////////////////////////////////////////////////

   void AssignDrawable(RDrawable *drawable, const std::string &prefix);

   void AssignParent(const RAttrBase *parent, const std::string &prefix);

   struct Rec_t {
      RAttrMap *attr{nullptr};
      std::string fullname;
      RDrawable *drawable{nullptr};
      operator bool() const { return !!attr; }
   };

   /// Find attributes container and full-qualified name for value
   const Rec_t AccessAttr(const std::string &name) const
   {
      const RAttrBase *prnt = this;
      std::string fullname = name;
      while (prnt) {
         fullname.insert(0, prnt->fPrefix); // fullname = prnt->fPrefix + fullname
         if (prnt->fDrawable)
            return {&(prnt->fDrawable->fAttr), fullname, prnt->fDrawable};
         if (prnt->fOwnAttr)
            return {prnt->fOwnAttr.get(), fullname, nullptr};
         prnt = prnt->fParent;
      }
      return {nullptr, fullname, nullptr};
   }

   struct Val_t {
      const RAttrMap::Value_t *value{nullptr};
      std::shared_ptr<RStyle> style;
      operator bool() const { return !!value; }
   };

   const Val_t AccessValue(const std::string &name, bool use_style = true) const
   {
      if (auto access = AccessAttr(name)) {
         if (auto rec = access.attr->Find(access.fullname))
            return {rec, nullptr};
         if (access.drawable && use_style)
            if (auto observe = access.drawable->fStyle.lock()) {
               if (auto rec = observe->Eval(access.fullname, access.drawable))
                  return {rec, observe};
            }
      }

      return {nullptr, nullptr};
   }

   Rec_t EnsureAttr(const std::string &name)
   {
      const RAttrBase *prnt = this;
      std::string fullname = name;
      while (prnt) {
         fullname.insert(0, prnt->fPrefix); // fullname = prnt->fPrefix + fullname
         if (prnt->fDrawable)
            return {&(prnt->fDrawable->fAttr), fullname, prnt->fDrawable};
         if (!prnt->fParent && !prnt->fOwnAttr)
            const_cast<RAttrBase *>(prnt)->fOwnAttr = std::make_unique<RAttrMap>();
         if (prnt->fOwnAttr)
            return {prnt->fOwnAttr.get(), fullname, nullptr};
         prnt = prnt->fParent;
      }
      return {nullptr, fullname, nullptr};
   }

   /// Evaluate attribute value

   template <typename RET_TYPE,typename MATCH_TYPE = void>
   auto Eval(const std::string &name, bool use_dflts = true) const
   {
      if (auto v = AccessValue(name, true))
         return RAttrMap::Value_t::get_value<RET_TYPE,MATCH_TYPE>(v.value);

      const RAttrMap::Value_t *rec = nullptr;

      if (use_dflts)
         rec = GetDefaults().Find(name);

      return RAttrMap::Value_t::get_value<RET_TYPE,MATCH_TYPE>(rec);
   }


   double *GetDoublePtr(const std::string &name) const;

   void CopyTo(RAttrBase &tgt, bool use_style = true) const;

   bool IsSame(const RAttrBase &src, bool use_style = true) const;

   RAttrBase(RDrawable *drawable, const std::string &prefix) { AssignDrawable(drawable, prefix); }

   RAttrBase(const RAttrBase *parent, const std::string &prefix) { AssignParent(parent, prefix); }

   RAttrBase(const RAttrBase &src) { src.CopyTo(*this); }

   RAttrBase &operator=(const RAttrBase &src)
   {
      Clear();
      src.CopyTo(*this);
      return *this;
   }

   void SetValue(const std::string &name, double value);
   void SetValue(const std::string &name, int value);
   void SetValue(const std::string &name, const std::string &value);

   const std::string &GetPrefix() const { return fPrefix; }

   void ClearValue(const std::string &name);

   void Clear();

   template <typename T = void>
   bool HasValue(const std::string &name, bool check_defaults = false) const
   {
      return Eval<const RAttrMap::Value_t *, T>(name, check_defaults) != nullptr;
   }

   template <typename T>
   T GetValue(const std::string &name) const
   {
      return Eval<T>(name);
   }

public:
   RAttrBase() = default;

   virtual ~RAttrBase() = default;

   friend bool operator==(const RAttrBase& lhs, const RAttrBase& rhs){ return lhs.IsSame(rhs) && rhs.IsSame(lhs); }
   friend bool operator!=(const RAttrBase& lhs, const RAttrBase& rhs){ return !lhs.IsSame(rhs) || !rhs.IsSame(lhs); }
};


} // namespace Experimental
} // namespace ROOT

#define R__ATTR_CLASS(ClassName,dflt_prefix,dflt_values) \
protected: \
const RAttrMap &GetDefaults() const override \
{ \
   static auto dflts = RAttrMap().dflt_values; \
   return dflts; \
} \
public: \
   ClassName() = default; \
   ClassName(RDrawable *drawable, const std::string &prefix = dflt_prefix) { AssignDrawable(drawable, prefix); } \
   ClassName(const RAttrBase *parent, const std::string &prefix = dflt_prefix) { AssignParent(parent, prefix); } \
   ClassName(const ClassName &src) : ClassName() { src.CopyTo(*this); } \
   ClassName &operator=(const ClassName &src) \
   { \
      Clear(); \
      src.CopyTo(*this); \
      return *this; \
   }

#endif
