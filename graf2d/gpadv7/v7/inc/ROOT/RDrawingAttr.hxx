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

class RDrawingAttrBase;
class RDrawingOptsBase;

/// \[ \name Attribute Stringification
float FromAttributeString(const std::string &strval, const std::string &name, float *);
double FromAttributeString(const std::string &strval, const std::string &name, double *);
char FromAttributeString(const std::string &strval, const std::string &name, char *);
short FromAttributeString(const std::string &strval, const std::string &name, short *);
int FromAttributeString(const std::string &strval, const std::string &name, int *);
long FromAttributeString(const std::string &strval, const std::string &name, long *);
long long FromAttributeString(const std::string &strval, const std::string &name, long long *);
unsigned char FromAttributeString(const std::string &strval, const std::string &name, unsigned char *);
unsigned short FromAttributeString(const std::string &strval, const std::string &name, unsigned short *);
unsigned int FromAttributeString(const std::string &strval, const std::string &name, unsigned int *);
unsigned long FromAttributeString(const std::string &strval, const std::string &name, unsigned long *);
unsigned long long FromAttributeString(const std::string &strval, const std::string &name, unsigned long long *);

/// Decode an enum value from its integer representation.
template <typename ENUM, class = typename std::enable_if<std::is_enum<ENUM>::value>::type>
ENUM FromAttributeString(const std::string &strval, const std::string &name, ENUM *)
{
   return static_cast<ENUM>(FromAttributeString(strval, name, (typename std::underlying_type<ENUM>::type*)nullptr));
}


std::string ToAttributeString(float val);
std::string ToAttributeString(double val);
std::string ToAttributeString(char val);
std::string ToAttributeString(short val);
std::string ToAttributeString(int val);
std::string ToAttributeString(long val);
std::string ToAttributeString(long long val);
std::string ToAttributeString(unsigned char val);
std::string ToAttributeString(unsigned short val);
std::string ToAttributeString(unsigned int val);
std::string ToAttributeString(unsigned long val);
std::string ToAttributeString(unsigned long long val);

/// Stringify an enum value through its integer representation.
template <typename ENUM, class = typename std::enable_if<std::is_enum<ENUM>::value>::type>
std::string ToAttributeString(ENUM val)
{
   return ToAttributeString(static_cast<typename std::underlying_type<ENUM>::type>(val));
}

/// \]


class RDrawingAttrHolder;

/** \class ROOT::Experimental::RDrawingAttrBase
 A collection of graphics attributes, for instance everything describing a line:
 color, width, opacity and style.
 It has a name, so it can be found in the style.
 */
class RDrawingAttrBase {
public:
   /// An attribute name part, e.g. "line".
   struct Name {
      Name() = default;
      Name(const std::string &name): fStr(name) {}
      Name(std::string &&name): fStr(std::move(name)) {}
      Name(const char *name): fStr(name) {}

      std::string fStr;
   };
   /// Combination of names, e.g. "hist2d.box.line.width".
   struct Path {
      /// Path in its dotted form.
      std::string fStr;

      Path() = default;

      explicit Path(const std::string &str): fStr(str) {}

      explicit Path(std::string &&str): fStr(std::move(str)) {}

      void Append(const Name &name) {
         if (!fStr.empty())
            fStr += ".";
         fStr += name.fStr;
      }

      Path& operator+=(const Name &name) {
         Append(name);
         return *this;
      }

      Path operator+(const Name &name) const {
         Path ret(*this);
         ret += name;
         return ret;
      }

      bool operator==(const Path &rhs) const { return fStr == rhs.fStr; }
      bool operator!=(const Path &rhs) const { return !(*this == rhs); }

      const std::string Str() const { return fStr; }
   };

protected:
   /// The chain of attribute names, as used in style files.
   /// E.g. "hist1D.hist.box.line".
   Path fPath;

   /// The container of the attribute values.
   std::weak_ptr<RDrawingAttrHolder> fHolder;   ///<!   I/O not working anyway

protected:
   /// Get the attribute value as string, for a given attribute name.
   std::string GetValueString(const Path &path) const;

   /// Insert or update the attribute value identified by the valueIndex (in fValueNames)
   /// to the value `strVal`.
   void SetValueString(const Name &name, const std::string &strVal);

   /// Construct a default, unnamed, unconnected attribute.
   RDrawingAttrBase() = default;

   /// Return `true` if the attribute's value comes from the
   /// styles, i.e. through `RDrawingAttrHolder::GetAttrFromStyle()`, instead
   /// if from our `RDrawingAttrHolder` (i.e. explicitly set through `Set()`).
   bool IsFromStyle(const Path &path) const;

public:
   /// Construct as a copy.
   RDrawingAttrBase(const RDrawingAttrBase& other) = default;

   /// Construct as a moved-to.
   RDrawingAttrBase(RDrawingAttrBase&& other) = default;

   /// Construct a named attribute that does not have a parent; e.g.
   /// because it's the top-most attribute in a drawing option object.
   RDrawingAttrBase(const Name &name): fPath{name.fStr} {}

   /// Construct a named attribute that has a parent, e.g.
   /// because it's some line attribute of the histogram attributes.
   RDrawingAttrBase(const Name &name, const RDrawingAttrBase &parent);

   /// Tag type to disambiguate construction from options.
   struct FromOption_t {};
   static constexpr const FromOption_t FromOption{};
   /// Construct a top-most attribute from its holder.
   RDrawingAttrBase(FromOption_t, const Name &name, RDrawingOptsBase &opts);

   /// Construct a top-most attribute from its holder. If this is ambiguous, use the
   /// tag overload taking an `FromOption_t`.
   RDrawingAttrBase(const Name &name, RDrawingOptsBase &opts):
      RDrawingAttrBase(FromOption, name, opts) {}

   /// Copy-assign: this assigns the attribute values to this attribute, *without*
   /// changing the connected drawing options object / holder or attribute path!
   ///
   /// It gives value semantics to attributes:
   /// ```
   /// DrawingOpts1 o1;
   /// DrawingOpts2 o2;
   /// RAttrLine l1 = o1.DogLine();
   /// RAttrLine l2 = o2.CatLine();
   /// l1.SetWidth(42);
   /// l2 = l1;
   /// // Now o2.CatLine().GetWidth() is 42!
   /// ```
   RDrawingAttrBase &operator=(const RDrawingAttrBase& rhs);

   /// Return `true` if the attribute's value comes from the
   /// styles, i.e. through `RDrawingAttrHolder::GetAttrFromStyle()`, instead
   /// if from our `RDrawingAttrHolder` (i.e. explicitly set through `Set()`).
   bool IsFromStyle(const Name &name) const;

   /// Get the attribute value for an attribute value of type `T`.
   template <class T>
   T Get(const Name &name) const
   {
      Path path = fPath + name;
      auto strVal = GetValueString(path);
      return FromAttributeString(strVal, path.Str(), (T*)nullptr);
   }

   /// Insert or update the attribute value identified by `name` to the given value.
   template <class T>
   void Set(const Name &name, const T &val)
   {
      SetValueString(name, ToAttributeString(val));
   }

   /// Return the attribute names that lead to this attribute, starting
   /// with the topmost attribute, i.e. the parent that does not have a parent
   /// itself, down to the name of *this (the last entry in the vector).
   const Path &GetPath() const { return fPath; }

   /// Actual attribute holder.
   const std::weak_ptr<RDrawingAttrHolder> &GetHolderPtr() const { return fHolder; }

   /// Equality compare to other RDrawingAttrBase.
   /// They are equal if
   /// - the same set of attributes are custom set (versus are determined from the style), and
   /// - the values of all the custom set ones compare equal.
   /// The set of styles to be taken into account is not compared.
   bool operator==(const RDrawingAttrBase &other) const;

   /// Compare unequal to other RDrawingAttrBase. Returns the negated `operator==`.
   bool operator!=(const RDrawingAttrBase &other) const
   {
      return !(*this == other);
   }
};


/** \class ROOT::Experimental::RDrawingAttrHolder
 A container of (stringified) attributes for which values have been provided.
 */
class RDrawingAttrHolder {
public:
   using Path_t = RDrawingAttrBase::Path;
   using Map_t = std::unordered_map<std::string, std::string>;
private:
   /// Map attribute paths to their values.
   Map_t fAttrNameVals;

   /// Attribute style classes of these options that will be "summed" in order,
   /// e.g. {"trigger", "efficiency"} will look attributes up in the `RDrawingAttrHolderBase` base class,
   /// if not found using the "trigger" style class, and if not found in the "efficiency" style class.
   /// Implicitly and as final resort, the attributes from the "default" style class will be used.
   std::vector<std::string> fStyleClasses;

public:
   /// RDrawingAttrHolder using only the default style.
   RDrawingAttrHolder() = default;

   /// RDrawingAttrHolder with an ordered collection of styles taking precedence before the default style.
   RDrawingAttrHolder(const std::vector<std::string> &styleClasses): fStyleClasses(styleClasses) {}

   /// Get an attribute value as string, given its name path.
   std::string &At(const Path_t &path) { return fAttrNameVals[path.fStr]; }

   /// Get an attribute value as pointer to string, given its name path, or
   /// `nullptr` if the attribute does not exist.
   const std::string *AtIf(const Path_t &path) const;

   /// Get the (stringified) value of the named attribute from the Style.
   /// Return the empty string if no such value exists - which means that the attribute
   /// name is unknown even for the (implicit) default style!
   std::string GetAttrFromStyle(const Path_t &path);

   /// Equality compare the attributes within `path` to those of `other` within `otherpath`.
   /// Takes all sub-attributes contained in the respective paths (i.e. those starting with that path)
   /// into account. They compare equal if their set of (sub-)attributes and their respective values are equal.
   bool Equal(const RDrawingAttrHolder &other, const Path_t &thisPath, const Path_t &otherPath);

   /// Extract contained attributes for a given path (including sub-attributes); returns iterators
   /// to a subset of fAttrNameVals.
   std::vector<Map_t::const_iterator> GetAttributesInPath(const Path_t &path) const;

   /// Erase all custom set attributes for a given path (including sub-attributes).
   void EraseAttributesInPath(const Path_t &path);

   /// Copy attributes within otherPath into
   void CopyAttributesInPath(const Path_t &targetPath, const RDrawingAttrHolder &source, const Path_t &sourcePath);

   /// Get the attribute style classes of these options.
   const std::vector<std::string> &GetStyleClasses() const { return fStyleClasses; }

   /// Set the attribute style classes of these options.
   void SetStyleClasses(const std::vector<std::string> &styles) { fStyleClasses = styles; }
};




/// Only attributes, which should be inserted into RDrawable, probably should be just part of base class

class RAttributesVisitor;

class RDrawableAttributes {

   friend class RAttributesVisitor;

public:

   enum EValuesKind { kBool, kInt, kDouble, kString };

   class Value_t {
   public:
      Value_t() = default;
      virtual ~Value_t() = default;
      virtual EValuesKind Kind() const = 0;
      virtual bool Compatible(EValuesKind kind) const { return kind == Kind(); }
      virtual int GetInt() const { return 0; }
      virtual double GetDouble() const { return 0; }
      virtual std::string GetString() const { return ""; }
      virtual Value_t *Copy() const = 0;
   };

   class IntValue_t : public Value_t {
      int v; ///< integer value
   public:
      IntValue_t(int _v = 0) : v(_v) {}
      EValuesKind Kind() const final { return kInt; }
      int GetInt() const final { return v; }
      Value_t *Copy() const final { return new IntValue_t(v); }
   };

   class DoubleValue_t : public Value_t {
      double v; ///< double value
   public:
      DoubleValue_t(double _v = 0) : v(_v) {}
      EValuesKind Kind() const final { return kDouble; }
      double GetDouble() const final { return v; }
      Value_t *Copy() const final { return new DoubleValue_t(v); }
   };

   class StringValue_t : public Value_t {
      std::string v; ///< string value
   public:
      StringValue_t(const std::string _v = "") : v(_v) {}
      EValuesKind Kind() const final { return kString; }
      std::string GetString() const final { return v; }
      Value_t *Copy() const final { return new StringValue_t(v); }
   };

   class Map_t {
      // FIXME: due to ROOT-10306 only data member of such kind can be correctly stored by ROOT I/O
      // Once problem fixed, one could make this container a base class
      std::unordered_map<std::string, std::unique_ptr<Value_t>> m; ///< JSON_object
   public:
      Map_t() = default; ///< JSON_asbase - store as map object

      Map_t &Add(const std::string &name, Value_t *value) { if (value) m[name] = std::unique_ptr<Value_t>(value); return *this; }
      Map_t &AddInt(const std::string &name, int value) { m[name] = std::make_unique<IntValue_t>(value); return *this; }
      Map_t &AddDouble(const std::string &name, double value) { m[name] = std::make_unique<DoubleValue_t>(value); return *this; }
      Map_t &AddString(const std::string &name, const std::string &value) { m[name] = std::make_unique<StringValue_t>(value); return *this; }

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

      const Value_t *Eval(const std::string &name) const
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
   Map_t map;                    ///<
   Map_t *defaults{nullptr};     ///<! default values for server

public:

   RDrawableAttributes() = default;

   RDrawableAttributes(const std::string &_type) { type = _type; }

   ~RDrawableAttributes() {}
};

//////////////////////////////////////////////////////////////////////////

class RStyleNew {
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

   RDrawableAttributes *fAttr{nullptr};                            ///<! source for attributes
   std::unique_ptr<RDrawableAttributes> fOwnAttr;                  ///<! own instance when deep copy is created
   std::string fPrefix;                                            ///<! name prefix for all attributes values
   const RDrawableAttributes::Map_t *fDefaults{nullptr};           ///<! defaults values for this visitor
   std::shared_ptr<RStyleNew> fStyle;                              ///<! style used for evaluations
   RAttributesVisitor *fParent{nullptr};                           ///<! parent attributes, prefix applied to it

   std::string GetFullName(const std::string &name) const { return fPrefix + name; }

protected:

   /** Normally should be configured in constructor */
   void SetDefaults(const RDrawableAttributes::Map_t *dflts) { fDefaults = dflts; }

   /** use const char* - nullptr means no value found */
   const RDrawableAttributes::Value_t *Eval(const std::string &name, bool use_dflts = true) const;

   void CreateOwnAttr();

   void AssignAttributes(RDrawableAttributes &cont, const std::string &prefix)
   {
      fAttr = &cont;
      fOwnAttr.reset();
      fPrefix = prefix;
      fParent = nullptr;
   }

   void AssignParent(RAttributesVisitor *parent, const std::string &prefix)
   {
      fAttr = nullptr;  // first access to attributes will chained to parent
      fOwnAttr.reset();
      fPrefix = prefix;
      fParent = parent;
   }

   bool GetAttr(bool force = false) const;

public:

   RAttributesVisitor() = default;

   RAttributesVisitor(RDrawableAttributes &cont, const std::string &prefix = "") { AssignAttributes(cont, prefix); }

   RAttributesVisitor(RAttributesVisitor *parent, const std::string &prefix = "") { AssignParent(parent, prefix); }

   RAttributesVisitor(const RAttributesVisitor &src) { fDefaults = src.fDefaults; CreateOwnAttr(); Copy(src); }

   RAttributesVisitor &operator=(const RAttributesVisitor &src) { Clear(); Copy(src); return *this; }

   void Copy(const RAttributesVisitor &src, bool use_dflts = false);

   void UseStyle(std::shared_ptr<RStyleNew> style) { fStyle = style; }

   /** returns true when value exists */
   bool HasValue(const std::string &name, bool use_dflts = false) const { return Eval(name, use_dflts) != nullptr; }

   void SetValue(const std::string &name, double value);
   void SetValue(const std::string &name, const int value);
   void SetValue(const std::string &name, const std::string &value);

   void ClearValue(const std::string &name);

   void Clear();

   std::string GetString(const std::string &name) const;
   int GetInt(const std::string &name) const;
   double GetDouble(const std::string &name) const;
};


} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingAttr
