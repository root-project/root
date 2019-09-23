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

class RDrawable;

// SL TODO: move all methods to protected area

/** Base class for all attributes, used with RDrawable */
class RAttrBase {

   friend class RAttrMap;

   RDrawable *fDrawable{nullptr};                          ///<! drawable used to store attributes
   RAttrMap *fAttr{nullptr};                               ///<! attributes sources
   std::unique_ptr<RAttrMap> fOwnAttr;                     ///<! own instance when deep copy is created
   std::string fPrefix;                                    ///<! name prefix for all attributes values
   const RAttrBase *fParent{nullptr};                      ///<! parent attributes, prefix applied to it

   std::string GetFullName(const std::string &name) const { return fPrefix + name; }

   std::string GetPrefixToParent() const;

protected:

   virtual const RAttrMap &GetDefaults() const
   {
      static RAttrMap empty;
      return empty;
   }

   bool CopyValue(const std::string &name, const RAttrMap::Value_t *value, bool check_type = true);

   bool IsValueEqual(const std::string &name, const RAttrMap::Value_t *value, bool use_style = false) const;

   ///////////////////////////////////////////////////////////////////////////////
   /// Evaluate attribute value

   template <typename T,typename S = void>
   auto Eval(const std::string &name, bool use_dflts = true) const
   {
      const RAttrMap::Value_t *rec = nullptr;

      if (GetAttr()) {
         auto fullname = GetFullName(name);

         rec = fAttr->Find(fullname);
         if (rec) return RAttrMap::Value_t::get_value<T,S>(rec);

         if (fDrawable)
           if (auto observe = fDrawable->fStyle.lock()) {
              rec = observe->Eval(fullname, fDrawable);
              if (rec) return RAttrMap::Value_t::get_value<T,S>(rec);
           }
      }

      if (use_dflts)
         rec = GetDefaults().Find(name);

      return RAttrMap::Value_t::get_value<T,S>(rec);
   }

   void CreateOwnAttr();

   void AssignDrawable(RDrawable *drawable, const std::string &prefix);

   void AssignParent(const RAttrBase *parent, const std::string &prefix);

   bool GetAttr() const;

   bool EnsureAttr();

   double *GetDoublePtr(const std::string &name) const;

   void SemanticCopy(const RAttrBase &src);

   bool IsSame(const RAttrBase &src, bool use_style = true) const;

public:

   RAttrBase() = default;

   RAttrBase(RDrawable *drawable, const std::string &prefix = "") { AssignDrawable(drawable, prefix); }

   RAttrBase(const RAttrBase *parent, const std::string &prefix = "") { AssignParent(parent, prefix); }

   RAttrBase(const RAttrBase &src) { src.CopyTo(*this); }

   virtual ~RAttrBase() = default;

   RAttrBase &operator=(const RAttrBase &src) { Clear(); src.CopyTo(*this); return *this; }

   void CopyTo(RAttrBase &tgt, bool use_style = true) const;

   void SetValue(const std::string &name, double value);
   void SetValue(const std::string &name, const int value);
   void SetValue(const std::string &name, const std::string &value);

   std::string GetPrefix() const { return fPrefix; }

   void ClearValue(const std::string &name);

   void Clear();

   template<typename T = void, std::enable_if_t<!std::is_pointer<T>{}>* = nullptr>
   bool HasValue(const std::string &name, bool check_defaults = false) const { return Eval<const RAttrMap::Value_t *,T>(name, check_defaults) != nullptr; }

   template<typename T, std::enable_if_t<!std::is_pointer<T>{}>* = nullptr>
   T GetValue(const std::string &name) const { return Eval<T>(name); }

   friend bool operator==(const RAttrBase& lhs, const RAttrBase& rhs){ return lhs.IsSame(rhs) && rhs.IsSame(lhs); }
   friend bool operator!=(const RAttrBase& lhs, const RAttrBase& rhs){ return !lhs.IsSame(rhs) || !rhs.IsSame(lhs); }
};


} // namespace Experimental
} // namespace ROOT

#define R_ATTR_CLASS(ClassName,dflt_prefix,dflt_values) \
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
   ClassName &operator=(const ClassName &src) { Clear(); src.CopyTo(*this); return *this; }

#endif
