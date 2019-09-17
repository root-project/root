/// \file ROOT/RDrawingAttr.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDrawingAttr
#define ROOT7_RDrawingAttr

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <list>

namespace ROOT {
namespace Experimental {

class RAttributesVisitor;

class RDrawableAttributes {

   friend class RAttributesVisitor;

public:

   enum EValuesKind { kBool, kInt, kDouble, kString };

   class Value_t {
      friend class RDrawableAttributes;
   public:
      Value_t() = default;
      virtual ~Value_t() = default;
      virtual EValuesKind Kind() const = 0;
      virtual bool Compatible(EValuesKind kind) const { return kind == Kind(); }
      virtual bool GetBool() const { return false; }
      virtual int GetInt() const { return 0; }
      virtual double GetDouble() const { return 0; }
      virtual std::string GetString() const { return ""; }
      virtual bool IsEqual(const Value_t *) const { return false; }
      virtual Value_t *Copy() const = 0;
      virtual const void *GetValuePtr() const { return nullptr; }

      template<typename T> T get() const;

      template <typename T, typename second = void>
      static T get_value(const Value_t *rec);
   };

   class BoolValue_t : public Value_t {
      bool v{false}; ///< integer value
   public:
      explicit BoolValue_t(bool _v = false) : v(_v) {}
      EValuesKind Kind() const final { return kBool; }
      bool GetBool() const final { return v; }
      Value_t *Copy() const final { return new BoolValue_t(v); }
      bool IsEqual(const Value_t *tgt) const final { return (tgt->Kind() == kBool) && (tgt->GetBool() == v); }
   };


   class IntValue_t : public Value_t {
      int v{0}; ///< integer value
   public:
      IntValue_t(int _v = 0) : v(_v) {}
      EValuesKind Kind() const final { return kInt; }
      int GetInt() const final { return v; }
      Value_t *Copy() const final { return new IntValue_t(v); }
      bool IsEqual(const Value_t *tgt) const final { return (tgt->Kind() == kInt) && (tgt->GetInt() == v); }
   };

   class DoubleValue_t : public Value_t {
      double v{0}; ///< double value
   public:
      DoubleValue_t(double _v = 0) : v(_v) {}
      EValuesKind Kind() const final { return kDouble; }
      double GetDouble() const final { return v; }
      Value_t *Copy() const final { return new DoubleValue_t(v); }
      bool IsEqual(const Value_t *tgt) const final { return (tgt->Kind() == kDouble) && (tgt->GetDouble() == v); }
      const void *GetValuePtr() const final { return &v; }
   };

   class StringValue_t : public Value_t {
      std::string v; ///< string value
   public:
      StringValue_t(const std::string _v = "") : v(_v) {}
      EValuesKind Kind() const final { return kString; }
      std::string GetString() const final { return v; }
      bool IsEqual(const Value_t *tgt) const final { return (tgt->Kind() == kString) && (tgt->GetString() == v); }
      Value_t *Copy() const final { return new StringValue_t(v); }
   };

   class Map_t {
      // FIXME: due to ROOT-10306 only data member of such kind can be correctly stored by ROOT I/O
      // Once problem fixed, one could make this container a base class
      std::unordered_map<std::string, std::unique_ptr<Value_t>> m; ///< JSON_object
   public:
      Map_t() = default; ///< JSON_asbase - store as map object

      Map_t &Add(const std::string &name, Value_t *value) { if (value) m[name] = std::unique_ptr<Value_t>(value); return *this; }
      Map_t &AddBool(const std::string &name, bool value) { m[name] = std::make_unique<BoolValue_t>(value); return *this; }
      Map_t &AddInt(const std::string &name, int value) { m[name] = std::make_unique<IntValue_t>(value); return *this; }
      Map_t &AddDouble(const std::string &name, double value) { m[name] = std::make_unique<DoubleValue_t>(value); return *this; }
      Map_t &AddString(const std::string &name, const std::string &value) { m[name] = std::make_unique<StringValue_t>(value); return *this; }
      Map_t &AddDefaults(const RAttributesVisitor &vis);

      double *GetDoublePtr(const std::string &name) const
      {
         auto pair = m.find(name);
         return ((pair != m.end()) && (pair->second->Kind() == kDouble)) ? (double *) pair->second->GetValuePtr() : nullptr;
      }

      Map_t(const Map_t &src)
      {
         for (const auto &pair : src.m)
            m.emplace(pair.first, std::unique_ptr<Value_t>(pair.second->Copy()));
      }

      Map_t &operator=(const Map_t &src)
      {
         m.clear();
         for (const auto &pair : src.m)
            m.emplace(pair.first, std::unique_ptr<Value_t>(pair.second->Copy()));
         return *this;
      }

      const Value_t *Find(const std::string &name) const
      {
         auto entry = m.find(name);
         return (entry != m.end()) ? entry->second.get() : nullptr;
      }

      void Clear(const std::string &name)
      {
         auto entry = m.find(name);
         if (entry != m.end())
            m.erase(entry);
      }

      auto begin() const { return m.begin(); }
      auto end() const { return m.end(); }
   };

private:
   std::string type;             ///<! drawable type, not stored in the root file, must be initialized
   std::string user_class;       ///<  user defined drawable class, can later go inside map
   Map_t map;                    ///<  central values storage

public:

   RDrawableAttributes() = default;

   RDrawableAttributes(const std::string &_type) { type = _type; }

   ~RDrawableAttributes() {}
};

template<> bool RDrawableAttributes::Value_t::get<bool>() const;
template<> int RDrawableAttributes::Value_t::get<int>() const;
template<> double RDrawableAttributes::Value_t::get<double>() const;
template<> std::string RDrawableAttributes::Value_t::get<std::string>() const;

template<> bool RDrawableAttributes::Value_t::get_value<bool,void>(const Value_t *rec);
template<> int RDrawableAttributes::Value_t::get_value<int,void>(const Value_t *rec);
template<> double RDrawableAttributes::Value_t::get_value<double,void>(const Value_t *rec);
template<> std::string RDrawableAttributes::Value_t::get_value<std::string,void>(const Value_t *rec);
template<> const RDrawableAttributes::Value_t *RDrawableAttributes::Value_t::get_value<const RDrawableAttributes::Value_t *,void>(const Value_t *rec);
template<> const RDrawableAttributes::Value_t *RDrawableAttributes::Value_t::get_value<const RDrawableAttributes::Value_t *,bool>(const Value_t *rec);
template<> const RDrawableAttributes::Value_t *RDrawableAttributes::Value_t::get_value<const RDrawableAttributes::Value_t *,int>(const Value_t *rec);
template<> const RDrawableAttributes::Value_t *RDrawableAttributes::Value_t::get_value<const RDrawableAttributes::Value_t *,double>(const Value_t *rec);
template<> const RDrawableAttributes::Value_t *RDrawableAttributes::Value_t::get_value<const RDrawableAttributes::Value_t *,std::string>(const Value_t *rec);


//////////////////////////////////////////////////////////////////////////

class RStyle {
public:

   struct Block_t {
      std::string selector;
      RDrawableAttributes::Map_t map; ///<    container
      Block_t() = default;
      Block_t(const std::string &_selector) : selector(_selector) {}

      Block_t(const Block_t &) {} // dummy, should not be used, but appears in dictionary
      Block_t& operator=(const Block_t &) = delete;
   };

   const RDrawableAttributes::Value_t *Eval(const std::string &type, const std::string &user_class, const std::string &field) const;

   RDrawableAttributes::Map_t &AddBlock(const std::string &selector)
   {
      fBlocks.emplace_back(selector);
      return fBlocks.back().map;
   }

private:
   std::list<Block_t> fBlocks;  // use std::list to avoid calling of Block_t copy constructor
};


/** Access to drawable attributes, never should be stored */
class RAttributesVisitor {

   friend class RDrawableAttributes;

   RDrawableAttributes *fAttr{nullptr};                            ///<! source for attributes
   std::unique_ptr<RDrawableAttributes> fOwnAttr;                  ///<! own instance when deep copy is created
   std::string fPrefix;                                            ///<! name prefix for all attributes values
   std::weak_ptr<RStyle> fStyle;                                   ///<! style used for evaluations
   const RAttributesVisitor *fParent{nullptr};                     ///<! parent attributes, prefix applied to it

   std::string GetFullName(const std::string &name) const { return fPrefix + name; }

   std::string GetPrefixToParent() const;

protected:

   virtual const RDrawableAttributes::Map_t &GetDefaults() const
   {
      static RDrawableAttributes::Map_t empty;
      return empty;
   }

   bool CopyValue(const std::string &name, const RDrawableAttributes::Value_t *value, bool check_type = true);

   bool IsValueEqual(const std::string &name, const RDrawableAttributes::Value_t *value, bool use_style = false) const;

   ///////////////////////////////////////////////////////////////////////////////
   /// Evaluate attribute value

   template <typename T,typename S = void>
   auto Eval(const std::string &name, bool use_dflts = true) const
   {
      auto fullname = GetFullName(name);

      const RDrawableAttributes::Value_t *rec = nullptr;

      if (GetAttr()) {
         rec = fAttr->map.Find(fullname);
         if (rec) return RDrawableAttributes::Value_t::get_value<T,S>(rec);

         const auto *prnt = this;
         while (prnt) {
            if (auto observe = prnt->fStyle.lock()) {
               rec = observe->Eval(fAttr->type, fAttr->user_class, fullname);
               if (rec) return RDrawableAttributes::Value_t::get_value<T,S>(rec);
            }
            prnt = prnt->fParent;
         }
      }

      if (use_dflts)
         rec = GetDefaults().Find(name);

      return RDrawableAttributes::Value_t::get_value<T,S>(rec);
   }

   void CreateOwnAttr();

   void AssignAttributes(RDrawableAttributes &cont, const std::string &prefix)
   {
      fAttr = &cont;
      fOwnAttr.reset();
      fPrefix = prefix;
      fParent = nullptr;
   }

   void AssignParent(const RAttributesVisitor *parent, const std::string &prefix)
   {
      fAttr = nullptr;  // first access to attributes will chained to parent
      fOwnAttr.reset();
      fPrefix = prefix;
      fParent = parent;
   }

   bool GetAttr() const;

   bool EnsureAttr();

   double *GetDoublePtr(const std::string &name) const;

   void SemanticCopy(const RAttributesVisitor &src);

   bool IsSame(const RAttributesVisitor &src, bool use_style = true) const;

public:

   RAttributesVisitor() = default;

   RAttributesVisitor(RDrawableAttributes &cont, const std::string &prefix = "") { AssignAttributes(cont, prefix); }

   RAttributesVisitor(const RAttributesVisitor *parent, const std::string &prefix = "") { AssignParent(parent, prefix); }

   RAttributesVisitor(const RAttributesVisitor &src) { src.CopyTo(*this); }

   virtual ~RAttributesVisitor() = default;

   RAttributesVisitor &operator=(const RAttributesVisitor &src) { Clear(); src.CopyTo(*this); return *this; }

   void CopyTo(RAttributesVisitor &tgt, bool use_style = true) const;

   void UseStyle(const std::shared_ptr<RStyle> &style) { fStyle = style; }

   void SetValue(const std::string &name, double value);
   void SetValue(const std::string &name, const int value);
   void SetValue(const std::string &name, const std::string &value);

   std::string GetPrefix() const { return fPrefix; }

   void ClearValue(const std::string &name);

   void Clear();

   template<typename T = void, std::enable_if_t<!std::is_pointer<T>{}>* = nullptr>
   bool HasValue(const std::string &name, bool check_defaults = false) const { return Eval<const RDrawableAttributes::Value_t *,T>(name, check_defaults) != nullptr; }

   template<typename T, std::enable_if_t<!std::is_pointer<T>{}>* = nullptr>
   T GetValue(const std::string &name) const { return Eval<T>(name); }

   friend bool operator==(const RAttributesVisitor& lhs, const RAttributesVisitor& rhs){ return lhs.IsSame(rhs) && rhs.IsSame(lhs); }
   friend bool operator!=(const RAttributesVisitor& lhs, const RAttributesVisitor& rhs){ return !lhs.IsSame(rhs) || !rhs.IsSame(lhs); }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingAttr
