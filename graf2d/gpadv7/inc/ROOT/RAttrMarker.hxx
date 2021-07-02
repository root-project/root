/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrMarker
#define ROOT7_RAttrMarker

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrMarker
\ingroup GpadROOT7
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2018-10-12
\brief A marker attributes.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrMarker : public RAttrAggregation {

   R__ATTR_CLASS(RAttrMarker, "marker");

   enum EStyle {
      kNone = 0,
      kDot = 1,
      kPlus = 2,
      kStar = 3,
      kCircle = 4,
      kMultiply = 5,
      kFullDotSmall = 6,
      kFullDotMedium = 7,
      kFullDotLarge = 8,
      kFullCircle = 20,
      kFullSquare = 21,
      kFullTriangleUp = 22,
      kFullTriangleDown = 23,
      kOpenCircle = 24,
      kOpenSquare = 25,
      kOpenTriangleUp = 26,
      kOpenDiamond = 27,
      kOpenCross = 28,
      kFullStar = 29,
      kOpenStar = 30,
      kOpenTriangleDown = 32,
      kFullDiamond = 33,
      kFullCross = 34,
      kOpenDiamondCross = 35,
      kOpenSquareDiagonal = 36,
      kOpenThreeTriangles = 37,
      kOctagonCross = 38,
      kFullThreeTriangles = 39,
      kOpenFourTrianglesX = 40,
      kFullFourTrianglesX = 41,
      kOpenDoubleDiamond = 42,
      kFullDoubleDiamond = 43,
      kOpenFourTrianglesPlus = 44,
      kFullFourTrianglesPlus = 45,
      kOpenCrossX = 46,
      kFullCrossX = 47,
      kFourSquaresX = 48,
      kFourSquaresPlus = 49
   };

public:

   RAttrValue<RColor> color{this, "color", RColor::kBlack}; ///<! marker color
   RAttrValue<double> size{this, "size", 1.};               ///<! marker size
   RAttrValue<int> style{this, "style", 1};                 ///<! marker style

   RAttrMarker(const RColor &_color, double _size, int _style) : RAttrMarker()
   {
      color = _color;
      size = _size;
      style = _style;
   }

};

} // namespace Experimental
} // namespace ROOT

#endif
