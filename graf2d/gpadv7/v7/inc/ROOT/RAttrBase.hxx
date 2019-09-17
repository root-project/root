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

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RStyle.hxx>

namespace ROOT {
namespace Experimental {


/** Base class for all attributes, used with RDrawable */
class RAttrBase {

   friend class RDrawingAttr;

   RDrawingAttr *fAttr{nullptr};                            ///<! source for attributes
   std::unique_ptr<RDrawingAttr> fOwnAttr;                  ///<! own instance when deep copy is created
   std::string fPrefix;                                            ///<! name prefix for all attributes values
   std::weak_ptr<RStyle> fStyle;                                   ///<! style used for evaluations
   const RAttrBase *fParent{nullptr};                              ///<! parent attributes, prefix applied to it

   std::string GetFullName(const std::string &name) const { return fPrefix + name; }

   std::string GetPrefixToParent() const;

protected:

   virtual const RDrawingAttr::Map_t &GetDefaults() const
   {
      static RDrawingAttr::Map_t empty;
      return empty;
   }

   bool CopyValue(const std::string &name, const RDrawingAttr::Value_t *value, bool check_type = true);

   bool IsValueEqual(const std::string &name, const RDrawingAttr::Value_t *value, bool use_style = false) const;

   ///////////////////////////////////////////////////////////////////////////////
   /// Evaluate attribute value

   template <typename T,typename S = void>
   auto Eval(const std::string &name, bool use_dflts = true) const
   {
      auto fullname = GetFullName(name);

      const RDrawingAttr::Value_t *rec = nullptr;

      if (GetAttr()) {
         rec = fAttr->map.Find(fullname);
         if (rec) return RDrawingAttr::Value_t::get_value<T,S>(rec);

         const auto *prnt = this;
         while (prnt) {
            if (auto observe = prnt->fStyle.lock()) {
               rec = observe->Eval(fAttr->type, fAttr->user_class, fullname);
               if (rec) return RDrawingAttr::Value_t::get_value<T,S>(rec);
            }
            prnt = prnt->fParent;
         }
      }

      if (use_dflts)
         rec = GetDefaults().Find(name);

      return RDrawingAttr::Value_t::get_value<T,S>(rec);
   }

   void CreateOwnAttr();

   void AssignAttributes(RDrawingAttr &cont, const std::string &prefix)
   {
      fAttr = &cont;
      fOwnAttr.reset();
      fPrefix = prefix;
      fParent = nullptr;
   }

   void AssignParent(const RAttrBase *parent, const std::string &prefix)
   {
      fAttr = nullptr;  // first access to attributes will chained to parent
      fOwnAttr.reset();
      fPrefix = prefix;
      fParent = parent;
   }

   bool GetAttr() const;

   bool EnsureAttr();

   double *GetDoublePtr(const std::string &name) const;

   void SemanticCopy(const RAttrBase &src);

   bool IsSame(const RAttrBase &src, bool use_style = true) const;

public:

   RAttrBase() = default;

   RAttrBase(RDrawingAttr &cont, const std::string &prefix = "") { AssignAttributes(cont, prefix); }

   RAttrBase(const RAttrBase *parent, const std::string &prefix = "") { AssignParent(parent, prefix); }

   RAttrBase(const RAttrBase &src) { src.CopyTo(*this); }

   virtual ~RAttrBase() = default;

   RAttrBase &operator=(const RAttrBase &src) { Clear(); src.CopyTo(*this); return *this; }

   void CopyTo(RAttrBase &tgt, bool use_style = true) const;

   void UseStyle(const std::shared_ptr<RStyle> &style) { fStyle = style; }

   void SetValue(const std::string &name, double value);
   void SetValue(const std::string &name, const int value);
   void SetValue(const std::string &name, const std::string &value);

   std::string GetPrefix() const { return fPrefix; }

   void ClearValue(const std::string &name);

   void Clear();

   template<typename T = void, std::enable_if_t<!std::is_pointer<T>{}>* = nullptr>
   bool HasValue(const std::string &name, bool check_defaults = false) const { return Eval<const RDrawingAttr::Value_t *,T>(name, check_defaults) != nullptr; }

   template<typename T, std::enable_if_t<!std::is_pointer<T>{}>* = nullptr>
   T GetValue(const std::string &name) const { return Eval<T>(name); }

   friend bool operator==(const RAttrBase& lhs, const RAttrBase& rhs){ return lhs.IsSame(rhs) && rhs.IsSame(lhs); }
   friend bool operator!=(const RAttrBase& lhs, const RAttrBase& rhs){ return !lhs.IsSame(rhs) || !rhs.IsSame(lhs); }
};

} // namespace Experimental
} // namespace ROOT

#endif
