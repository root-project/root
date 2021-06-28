/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrText
#define ROOT7_RAttrText

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrText
\ingroup GpadROOT7
\brief A text attributes.
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrText : public RAttrAggregation {

   R__ATTR_CLASS(RAttrText, "text");

   RAttrValue<RColor> color{this, "color", RColor::kBlack};  ///<! text color
   RAttrValue<double> size{this, "size", 12.};               ///<! text size
   RAttrValue<double> angle{this, "angle", 0.};              ///<! text angle
   RAttrValue<int> align{this, "align", 22};                 ///<! text align
   RAttrValue<std::string> font_family{this, "font_family"};  ///<! font family, corresponds to css font-familty attribute
   RAttrValue<std::string> font_style{this, "font_style"};    ///<! font style, corresponds to css font-style attribute
   RAttrValue<std::string> font_weight{this, "font_weight"};  ///<! font weight, corresponds to css font-weight attribute

   ///Set text font by id as usually handled in the ROOT (without precision), number should be between 1 and 15
   RAttrText &SetFont(int font)
   {
      switch(font) {
      case 1: font_family = "Times New Roman"; font_style = "italic"; break;
      case 2: font_family = "Times New Roman"; font_weight = "bold"; break;
      case 3: font_family = "Times New Roman"; font_style = "italic"; font_weight = "bold"; break;
      case 4: font_family = "Arial"; break;
      case 5: font_family = "Arial"; font_style = "oblique"; break;
      case 6: font_family = "Arial"; font_weight = "bold"; break;
      case 7: font_family = "Arial"; font_style = "oblique"; font_weight = "bold"; break;
      case 8: font_family = "Courier New"; break;
      case 9: font_family = "Courier New"; font_style = "oblique"; break;
      case 10: font_family = "Courier New"; font_weight = "bold"; break;
      case 11: font_family = "Courier New"; font_style = "oblique"; font_weight = "bold"; break;
      case 12: font_family = "Symbol"; break;
      case 13: font_family = "Times New Roman"; break;
      case 14: font_family = "Wingdings"; break;
      case 15: font_family = "Symbol"; font_style = "italic"; break;
      }
      return *this;
   }

};



} // namespace Experimental
} // namespace ROOT

#endif
