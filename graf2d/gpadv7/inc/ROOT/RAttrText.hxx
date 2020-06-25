/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrText
#define ROOT7_RAttrText

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RAttrColor.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrText
\ingroup GpadROOT7
\brief A text.attributes.
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrText : public RAttrBase {

   RAttrColor           fColor{this, "color_"};       ///<! text color
   RAttrValue<double>   fSize{this, "size", 12.};     ///<! text size
   RAttrValue<double>   fAngle{this, "angle", 0.};    ///<! text angle
   RAttrValue<int>      fAlign{this, "align", 22};    ///<! text align
   RAttrValue<int>      fFont{this, "font", 41};      ///<! text font

   R__ATTR_CLASS(RAttrText, "text_", AddDefaults(fColor).AddDefaults(fSize).AddDefaults(fAngle).AddDefaults(fAlign).AddDefaults(fFont));

   ///The text size
   RAttrText &SetSize(double sz) { fSize = sz; return *this; }
   double GetSize() const { return fSize; }

   ///The text angle
   RAttrText &SetAngle(double angle) { fAngle = angle; return *this; }
   double GetAngle() const { return fAngle; }

   ///The text alignment
   RAttrText &SetAlign(int align) { fAlign = align; return *this; }
   int GetAlign() const { return fAlign; }

   ///The text font
   RAttrText &SetFont(int font) { fFont = font; return *this; }
   int GetFont() const { return fFont; }

   ///The color of the text.
   RAttrText &SetColor(const RColor &color) { fColor = color; return *this; }
   RColor GetColor() const { return fColor.GetColor(); }
   RAttrColor &AttrColor() { return fColor; }

};



} // namespace Experimental
} // namespace ROOT

#endif
