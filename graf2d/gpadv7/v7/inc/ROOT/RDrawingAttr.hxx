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

#include <functional>
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
inline std::string FromAttributeString(const std::string &strval, const std::string & /*name*/, std::string *)
{
   return strval;
}

/// Decode an enum value from its integer representation.
template <typename ENUM, class = typename std::enable_if<std::is_enum<ENUM>::value>::type>
inline ENUM FromAttributeString(const std::string &strval, const std::string &name, ENUM *)
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
inline std::string ToAttributeString(const std::string &val) { return val; }

/// Stringify an enum value through its integer representation.
template <typename ENUM, class = typename std::enable_if<std::is_enum<ENUM>::value>::type>
inline std::string ToAttributeString(ENUM val)
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
      explicit Path(const char *name): fStr(name) {}
      explicit Path(const Name &name): fStr(name.fStr) {}
      explicit Path(Name &&name): fStr(std::move(name.fStr)) {}

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
   struct MemberAssociation {
      Name fName;
      RDrawingAttrBase *fNestedAttr;
      std::function<std::string()> fMemberToString;
      std::function<void(const std::string&, const std::string&)> fSetMemberFromString;
   };

protected:
   MemberAssociation Associate(const std::string &name, RDrawingAttrBase &attr) {
      return {name, &attr, {}, {}};
   }

   template <class VALUE, class = typename std::enable_if<!std::is_base_of<RDrawingAttrBase, VALUE>::value>::type>
   MemberAssociation Associate(const std::string &name, VALUE &val) {
      return {name,
         nullptr,
         [&val]{return ToAttributeString(val);},
         [&val](const std::string &strval, const std::string &attrname) {
            val = FromAttributeString(strval, attrname, &val); }
      };
   }

   /// Create the name / member association for attribute values.
   virtual std::vector<MemberAssociation> GetMembers() = 0;

   /// Construct a default, unnamed, unconnected attribute.
   RDrawingAttrBase() = default;

   /// Initialize this from a style's values.
   void InitializeFromStyle(const Path &path, const RDrawingAttrHolder &attr);

public:
   virtual ~RDrawingAttrBase() = default;

   /// Insert all attribute members' values into keyval, using their name as the first string
   /// and their stringified value as the second. This pair is only inserted into keyval if
   /// the attribute's value is different than value provided by the style.
   void InsertModifiedAttributeStrings(const Path &path, const RDrawingAttrHolder &holder,
                                       std::vector<std::pair<std::string, std::string>> &keyval);
};


/** \class ROOT::Experimental::RDrawingAttrHolder
 A container of (stringified) attributes for which values have been provided.
 */
class RDrawingAttrHolder {
public:
   using Name_t = RDrawingAttrBase::Name;
   using Path_t = RDrawingAttrBase::Path;
   using Map_t = std::unordered_map<std::string, std::unique_ptr<RDrawingAttrBase>>;
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

   /// Get an attribute value as pointer to string, given its name path, or
   /// `nullptr` if the attribute does not exist.
   RDrawingAttrBase *AtIf(const Name_t &path) const;

   template <class ATTR>
   ATTR &Insert(const Name_t &name)
   {
      auto &attrPtr = fAttrNameVals[name.fStr];
      attrPtr = std::make_unique<ATTR>();
      return *static_cast<ATTR*>(attrPtr.get());
   }

   /// Get the attribute style classes of these options.
   const std::vector<std::string> &GetStyleClasses() const { return fStyleClasses; }

   /// Set the attribute style classes of these options.
   void SetStyleClasses(const std::vector<std::string> &styles) { fStyleClasses = styles; }

   /// Retrieve the attribute string from the current style, given the set of style classes.
   std::string GetAttrValStringFromStyle(const Path_t &path) const;

   /// Stringify all attributes set to non-default values, returning a collection of [name, value] pairs.
   std::vector<std::pair<std::string, std::string>> CustomizedValuesToString(const Name_t &option_name);
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RDrawingAttr
