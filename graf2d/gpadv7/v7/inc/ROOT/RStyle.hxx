/// \file ROOT/RStyle.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-10-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RStyle
#define ROOT7_RStyle

#include <ROOT/RColor.hxx>

#include <ROOT/RStringView.hxx>

#include <string>
#include <tuple>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RStyle
  A set of defaults for graphics attributes, e.g. for histogram fill color, line width, frame offsets etc.
  */

class RStyle {
public:
   /// A map of attribute name to string attribute values/
   using Attrs_t = std::unordered_map<std::string, std::string>;

private:
   /// Mapping of user coordinates to normal coordinates.
   std::string fName; // Name of the attribute set.
   Attrs_t fAttrs; // Pairs of name / attribute values.

public:
   /// Default constructor, creating an unnamed, empty style.
   RStyle() = default;

   /// Creates a named but empty style.
   explicit RStyle(std::string_view name): fName(name) {}

   /// Constructor taking the style name and a set of attributes (e.g. read from the config files).
   RStyle(std::string_view name, Attrs_t &&attrs): fName(name), fAttrs(std::move(attrs)) {}

   /// Get this stryle's name. (No setter as that would upset the unordered_map.)
   const std::string &GetName() const { return fName; }

   /// Get the style value as a string, given an attribute name.
   /// Strips leading "foo." from the name until the first entry in the style is found.
   /// E.g. if the default style has not entry for "Hist.1D.Fill.Color", the value might
   /// be initialized to the default style's entry for the more general "1D.Fill.Color".
   /// The className is the name of a CSS-style class, possibly overriding the generic attribute.
   std::string GetAttribute(const std::string &attrName, const std::string &className = {}) const;

   /// Move-register `style` in the global style collection, possibly replacing a global style with the same name.
   static RStyle &Register(RStyle &&style);

   /// Get the `RStyle` named `name` from the global style collection, or `nullptr` if that doesn't exist.
   static RStyle *Get(std::string_view name);

   /// Get the current RStyle.
   static RStyle &GetCurrent();

   /// Set the current RStyle by copying `style` into the static current style object.
   static void SetCurrent(const RStyle &style) { GetCurrent() = style; }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RStyle
