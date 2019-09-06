/// \file ROOT/RAttrLine.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-10-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrLine
#define ROOT7_RAttrLine

#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RColor.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrLine
 Drawing attributes for RLine.
 */
class RAttrLine: public RDrawingAttrBase {
public:
   using RDrawingAttrBase::RDrawingAttrBase;

   /// The color of the line.
   RAttrLine &SetColor(const RColor& col) { Set("color", col); return *this; }
   RColor GetColor() const { return Get<RColor>("color"); }

   ///The width of the line.
   RAttrLine &SetWidth(float width) { Set("width", width); return *this; }
   float GetWidth() const { return Get<float>("width"); }

   ///The style of the line.
   RAttrLine &SetStyle(int style) { Set("style", style); return *this; }
   int GetStyle() const { return Get<int>("style"); }
};


class RAttrLineNew : public RAttributesVisitor {

   auto &getDefaults()
   {
      static RDrawableAttributes::Map_t dflts;
      if (dflts.empty()) {
         dflts["width"] = std::make_unique<RDrawableAttributes::DoubleValue_t>(1);
         dflts["style"] = std::make_unique<RDrawableAttributes::IntValue_t>(1);
         dflts["color"] = std::make_unique<RDrawableAttributes::StringValue_t>("white");
      }
      return dflts;
   }

public:

   RAttrLineNew(RDrawableAttributes &cont, const std::string &prefix = "line_") :
      RAttributesVisitor(cont, prefix)
   {
      SetDefaults(getDefaults());
   }

   RAttrLineNew(const RDrawableAttributes &cont, const std::string &prefix = "line_") :
      RAttributesVisitor(cont, prefix)
   {
      SetDefaults(getDefaults());
   }

   ///The width of the line.
   RAttrLineNew &SetWidth(double width) { SetValue("width", width); return *this; }
   double GetWidth() const { return GetDouble("width"); }

   ///The style of the line.
   RAttrLineNew &SetStyle(int style) { SetValue("style", style); return *this; }
   int GetStyle() const { return GetInt("style"); }

   ///The color of the line.
   RAttrLineNew &SetColor(const std::string &color) { SetValue("color", color); return *this; }
   std::string GetColor() const { return GetString("color"); }

};

} // namespace Experimental
} // namespace ROOT

#endif
