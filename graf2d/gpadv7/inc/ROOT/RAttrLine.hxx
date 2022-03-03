/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrLine
#define ROOT7_RAttrLine

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrLine
\ingroup GpadROOT7
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\brief Drawing line attributes for different objects.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrLine : public RAttrAggregation {

   R__ATTR_CLASS(RAttrLine, "line");

public:

   enum EStyle {
      kNone = 0,
      kSolid = 1,
      kDashed = 2,
      kDotted = 3,
      kDashDotted = 4,
      kStyle1 = 1,
      kStyle2 = 2,
      kStyle3 = 3,
      kStyle4 = 4,
      kStyle5 = 5,
      kStyle6 = 6,
      kStyle7 = 7,
      kStyle8 = 8,
      kStyle9 = 9,
      kStyle10 = 10
   };

   RAttrValue<RColor>  color{this, "color", RColor::kBlack}; ///<! line color
   RAttrValue<double>  width{this, "width", 1.};             ///<! line width
   RAttrValue<EStyle>  style{this, "style", kSolid};         ///<! line style
   RAttrValue<std::string> pattern{this, "pattern"};         ///<! line pattern like "3,2,3,1,5"

   RAttrLine(const RColor &_color, double _width, EStyle _style) : RAttrLine()
   {
      color = _color;
      width = _width;
      style = _style;
   }

};


/** \class RAttrLineEnding
\ingroup GpadROOT7
\author  Sergey Linev <s.linev@gsi.de>
\date 2021-06-28
\brief Attributes for line ending
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrLineEnding : public RAttrAggregation {

   R__ATTR_CLASS(RAttrLineEnding, "ending");

public:

   RAttrValue<std::string> style{this, "style", ""};       ///<! axis ending style - none, arrow, circle
   RAttrValue<RPadLength> size{this, "size", 0.02_normal}; ///<! ending size

   void SetArrow() { style = "arrow"; }
   void SetCircle() { style = "cicrle"; }
};


} // namespace Experimental
} // namespace ROOT

#endif
