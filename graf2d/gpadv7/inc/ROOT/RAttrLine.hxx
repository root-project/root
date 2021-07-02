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

   RAttrValue<RColor>  color{this, "color", RColor::kBlack}; ///<! line color
   RAttrValue<double>  width{this, "width", 1.};             ///<! line width
   RAttrValue<int>     style{this, "style", 1};              ///<! line style

   RAttrLine(const RColor &_color, double _width, int _style) : RAttrLine()
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
