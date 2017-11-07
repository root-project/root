/// \file ROOT/TStyle.hxx
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

#ifndef ROOT7_TStyle
#define ROOT7_TStyle

#include <ROOT/TColor.hxx>

#include <RStringView.h>

#include <string>
#include <tuple>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TStyle
  A set of defaults for graphics attributes, e.g. for histogram fill color, line width, frame offsets etc.
  */

class TStyle {
public:
   /// A map of attribute name to attribute value for an attribute type `PRIMITIVE`.
   template <class PRIMITIVE> using Attrs_t = std::unordered_map<std::string, PRIMITIVE>;
   using AllAttrs_t = std::tuple<Attrs_t<TColor>, Attrs_t<long long>, Attrs_t<double>>;   
      
private:
   /// Mapping of user coordinates to normal coordinates.
   std::string fName; // Name of the attribute set.
   AllAttrs_t fAttrs; // Pairs of name / attribute values.

public:
   /// Default constructor, creating an unnamed, empty style.
   TStyle() = default;

   /// Creates a named but empty style.
   explicit TStyle(std::string_view name): fName(name) {}

   /// Constructor taking user coordinate system, position and extent.
   TStyle(std::string_view name, AllAttrs_t&& attrs): fName(name), fAttrs(std::move(attrs)) {}

   const std::string& GetName() const { return fName; }

   /// Register a copy of `style` in the global style collection, possibly replacing a global style with the same name.
   static void Register(const TStyle& style);

   /// Get the `TStyle` named `name` from the global style collection, or `nullptr` if that doesn't exist.
   static TStyle *Get(std::string_view name);

   /// Get the current TStyle.
   static TStyle &GetCurrent();

   /// Set the current TStyle by copying `style` into the static current style object.
   static void SetCurrent(const TStyle& style)
   {
      GetCurrent() = style;      
   }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TStyle
