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
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrLine
\ingroup GpadROOT7
\author Axel Naumann <axel@cern.ch>
\date 2018-10-12
\brief Drawing line attributes for different objects.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrLine : public RAttrBase {

   RAttrValue<RColor>  fLineColor{"color", this, RColor::kBlack}; ///<! line color
   RAttrValue<double>  fLineWidth{"width", this, 1.};             ///<! line width
   RAttrValue<int>     fLineStyle{"style", this, 1};              ///<! line style

   R__ATTR_CLASS(RAttrLine, "line");

   ///The width of the line.
   RAttrLine &SetLineWidth(double width) { fLineWidth = width; return *this; }
   double GetLineWidth() const { return fLineWidth; }

   ///The style of the line.
   RAttrLine &SetLineStyle(int style) { fLineStyle = style; return *this; }
   int GetLineStyle() const { return fLineStyle; }

   ///The color of the line.
   RAttrLine &SetLineColor(const RColor &color) { fLineColor = color; return *this; }
   RColor GetLineColor() const { return fLineColor; }

   const RAttrLine &AttrLine() const { return *this; }
   RAttrLine &AttrLine() { return *this; }

};

} // namespace Experimental
} // namespace ROOT

#endif
