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

   enum EFont {
      kTimesItalic = 1,
      kTimesBold = 2,
      kTimesBoldItalic = 3,
      kArial = 4,
      kArialOblique = 5,
      kArialBold = 6,
      kArialBoldOblique = 7,
      kCourier = 8,
      kCourierOblique = 9,
      kCourierBold = 10,
      kCourierBoldOblique = 11,
      // kSymbol = 12, // not supported by web browsers
      kTimes = 13,
      // kWingdings = 14,
      // kSymbolItalic = 15, // not supported by web browsers
      kVerdana = 16,
      kVerdanaItalic = 17,
      kVerdanaBold = 18,
      kVerdanaBoldItalic = 19
   };

   ///Set text font by id as usually handled in the ROOT (without precision), number should be between 1 and 15
   RAttrFont &SetFont(EFont font)
   {
      switch(font) {
      case kTimesItalic: family = "Times New Roman"; style = "italic"; break;
      case kTimesBold: family = "Times New Roman"; weight = "bold"; break;
      case kTimesBoldItalic: family = "Times New Roman"; style = "italic"; weight = "bold"; break;
      case kArial: family = "Arial"; break;
      case kArialOblique: family = "Arial"; style = "oblique"; break;
      case kArialBold: family = "Arial"; weight = "bold"; break;
      case kArialBoldOblique: family = "Arial"; style = "oblique"; weight = "bold"; break;
      case kCourier: family = "Courier New"; break;
      case kCourierOblique: family = "Courier New"; style = "oblique"; break;
      case kCourierBold: family = "Courier New"; weight = "bold"; break;
      case kCourierBoldOblique: family = "Courier New"; style = "oblique"; weight = "bold"; break;
      // case kSymbol: family = "Symbol"; break;
      case kTimes: family = "Times New Roman"; break;
      // case kWingdings: family = "Wingdings"; break;
      // case kSymbolItalic: family = "Symbol"; style = "italic"; break;
      case kVerdana: family = "Verdana";  break;
      case kVerdanaItalic: family = "Verdana";  style = "italic";  break;
      case kVerdanaBold: family = "Verdana";  weight = "bold"; break;
      case kVerdanaBoldItalic: family = "Verdana";  weight = "bold"; style = "italic"; break;
      }
      return *this;
   }

   /// assign font id, setting all necessary properties
   RAttrFont &operator=(EFont id) { SetFont(id); return *this; }

   friend bool operator==(const RAttrFont &font, EFont id) { RAttrFont font2; font2.SetFont(id); return font == font2; }

   /// Returns full font name including weight and style
   std::string GetFullName() const
   {
      std::string name = family, s = style, w = weight;
      if (!w.empty()) {
         name += " ";
         name += w;
      }
      if (!s.empty()) {
         name += " ";
         name += s;
      }
      return name;
   }

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

   enum EAlign {
      kLeftBottom = 11,
      kLeftCenter = 12,
      kLeftTop = 13,
      kCenterBottom = 21,
      kCenter = 22,
      kCenterTop = 23,
      kRightBottom = 31,
      kRightCenter = 32,
      kRightTop = 33
   };

   RAttrValue<RColor> color{this, "color", RColor::kBlack};  ///<! text color
   RAttrValue<double> size{this, "size", 12.};               ///<! text size
   RAttrValue<double> angle{this, "angle", 0.};              ///<! text angle
   RAttrValue<EAlign> align{this, "align", kCenter};         ///<! text align
   RAttrFont font{this, "font"};                             ///<! text font

   RAttrText(RDrawable *drawable, const char *prefix, double _size) : RAttrAggregation(drawable, prefix), size(this, "size", _size) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
