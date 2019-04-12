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
   using Name_t = RDrawingAttrBase::Path;
private:
   using Map_t = std::unordered_map<std::string, std::string>;
   /// Map attribute names to their values.
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
   std::string &At(const Name_t &attrName) { return fAttrNameVals[attrName.fStr]; }

   /// Get an attribute value as pointer to string, given its name path, or
   /// `nullptr` if the attribute does not exist.
   const std::string *AtIf(const Name_t &attrName) const;

   /// Get the (stringified) value of the names attribute from the Style.
   /// Return the empty string if no such value exists - which means that the attribute
   /// name is unknown even for the (implicit) default style!
   std::string GetAttrFromStyle(const Name_t &attrName);

   /// Equality compare the attributes starting with `name` to those of `other` starting wityh `otherName`.
   /// Takes all sub-attributes (i.e. those starting with that name) into account.
   /// They compare equal if their set of (sub-)attributes and their respective values are equal.
   bool Equal(const RDrawingAttrHolder &other, const Name_t &thisName, const Name_t &otherName);

   /// Get the attribute style classes of these options.
   const std::vector<std::string> &GetStyleClasses() const { return fStyleClasses; }

   /// Set the attribute style classes of these options.
   void SetStyleClasses(const std::vector<std::string> &styles) { fStyleClasses = styles; }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingAttr
