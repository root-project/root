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


/** \class RAttrFont
\ingroup GpadROOT7
\brief A font attributes, used together with text attributes
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2021-06-28
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrFont : public RAttrAggregation {

   R__ATTR_CLASS(RAttrFont, "font");

public:

   RAttrValue<std::string> family{this, "family"};  ///<! font family, corresponds to css font-familty attribute
   RAttrValue<std::string> style{this, "style"};    ///<! font style, corresponds to css font-style attribute
   RAttrValue<std::string> weight{this, "weight"};  ///<! font weight, corresponds to css font-weight attribute

   ///Set text font by id as usually handled in the ROOT (without precision), number should be between 1 and 15
   RAttrFont &SetFont(int font)
   {
      switch(font) {
      case 1: family = "Times New Roman"; style = "italic"; break;
      case 2: family = "Times New Roman"; weight = "bold"; break;
      case 3: family = "Times New Roman"; style = "italic"; weight = "bold"; break;
      case 4: family = "Arial"; break;
      case 5: family = "Arial"; style = "oblique"; break;
      case 6: family = "Arial"; weight = "bold"; break;
      case 7: family = "Arial"; style = "oblique"; weight = "bold"; break;
      case 8: family = "Courier New"; break;
      case 9: family = "Courier New"; style = "oblique"; break;
      case 10: family = "Courier New"; weight = "bold"; break;
      case 11: family = "Courier New"; style = "oblique"; weight = "bold"; break;
      case 12: family = "Symbol"; break;
      case 13: family = "Times New Roman"; break;
      case 14: family = "Wingdings"; break;
      case 15: family = "Symbol"; style = "italic"; break;
      }
      return *this;
   }

   RAttrFont &operator=(int id) { SetFont(id); return *this; }

};


/** \class RAttrText
\ingroup GpadROOT7
\brief A text attributes.
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrText : public RAttrAggregation {

   R__ATTR_CLASS(RAttrText, "text");

public:

   RAttrValue<RColor> color{this, "color", RColor::kBlack};  ///<! text color
   RAttrValue<double> size{this, "size", 12.};               ///<! text size
   RAttrValue<double> angle{this, "angle", 0.};              ///<! text angle
   RAttrValue<int> align{this, "align", 22};                 ///<! text align
   RAttrFont font{this, "font"};                             ///<! text font

   RAttrText(RDrawable *drawable, const char *prefix, double _size) : RAttrAggregation(drawable, prefix), size(this, "size", _size) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
