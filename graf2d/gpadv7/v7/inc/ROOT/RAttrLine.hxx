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

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RColor.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrLine
 Drawing line attributes for different objects.
 */

class RAttrLine : public RAttrBase {

   RColor fColor{this, "color_"}; ///<! line color, will access container from line attributes

   R_ATTR_CLASS(RAttrLine, "line_", AddDouble("width",1.).AddInt("style",1).AddDefaults(fColor));

   // keep it here, it is minimal set of methods which should be reimplemented
   // using RAttrBase::RAttrBase;
   // RAttrLine(const RAttrLine &src) : RAttrLine() { src.CopyTo(*this); }
   // RAttrLine &operator=(const RAttrLine &src) { Clear(); src.CopyTo(*this); return *this; }

   ///The width of the line.
   RAttrLine &SetWidth(double width) { SetValue("width", width); return *this; }
   double GetWidth() const { return GetValue<double>("width"); }

   ///The style of the line.
   RAttrLine &SetStyle(int style) { SetValue("style", style); return *this; }
   int GetStyle() const { return GetValue<int>("style"); }

   ///The color of the line.
   RAttrLine &SetColor(const RColor &color) { fColor = color; return *this; }
   const RColor &Color() const { return fColor; }
   RColor &Color() { return fColor; }

};

} // namespace Experimental
} // namespace ROOT

#endif
