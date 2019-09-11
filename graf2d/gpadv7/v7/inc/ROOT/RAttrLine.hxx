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

   RColorNew fColor{this, "color_"}; ///<! line color, will access container from line attributes

protected:
   const RDrawableAttributes::Map_t &GetDefaults() const override
   {
      static auto dflts = RDrawableAttributes::Map_t().AddDouble("width",1.).AddInt("style",1);
      return dflts;
   }

public:

   RAttrLineNew() : RAttributesVisitor()
   {
   }

   RAttrLineNew(RDrawableAttributes &cont, const std::string &prefix = "line_") : RAttrLineNew()
   {
      AssignAttributes(cont, prefix);
   }

   ///The width of the line.
   RAttrLineNew &SetWidth(double width) { SetValue("width", width); return *this; }
   double GetWidth() const { return GetDouble("width"); }

   ///The style of the line.
   RAttrLineNew &SetStyle(int style) { SetValue("style", style); return *this; }
   int GetStyle() const { return GetInt("style"); }

   ///The color of the line.
   RAttrLineNew &SetColor(const RColorNew &color) { fColor = color; return *this; }
   const RColorNew &Color() const { return fColor; }
   RColorNew &Color() { return fColor; }

};

} // namespace Experimental
} // namespace ROOT

#endif
