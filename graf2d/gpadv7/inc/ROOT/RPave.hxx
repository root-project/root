/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPave
#define ROOT7_RPave

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {


/** \class ROOT::Experimental::RPave
\ingroup GrafROOT7
\brief Base class for paves with text, statistic, legends, placed relative to RFrame position and adjustable height
\author Sergey Linev <s.linev@gsi.de>
\date 2020-06-18
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RPave : public RDrawable, public RAttrText, public RAttrLine, public RAttrFill {

   RAttrValue<RPadLength> fCornerX{this, "cornerx", 0.02};  ///<! X corner
   RAttrValue<RPadLength> fCornerY{this, "cornery", 0.02};  ///<! Y corner
   RAttrValue<RPadLength> fWidth{this, "width", 0.4};       ///<! pave width
   RAttrValue<RPadLength> fHeight{this, "height", 0.2};     ///<! pave height

public:

   RPave(const std::string &csstype = "pave") : RDrawable(csstype), RAttrText(this), RAttrLine(this), RAttrFill(this) {}

   RPave &SetCornerX(const RPadLength &pos) { fCornerX = pos; return *this; }
   RPadLength GetCornerX() const { return fCornerX; }

   RPave &SetCornerY(const RPadLength &pos) { fCornerY = pos; return *this; }
   RPadLength GetCornerY() const { return fCornerY; }

   RPave &SetWidth(const RPadLength &width) { fWidth = width; return *this; }
   RPadLength GetWidth() const { return fWidth; }

   RPave &SetHeight(const RPadLength &height) { fHeight = height; return *this; }
   RPadLength GetHeight() const { return fHeight; }
};

} // namespace Experimental
} // namespace ROOT

#endif
