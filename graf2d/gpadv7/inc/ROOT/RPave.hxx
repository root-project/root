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
#include <ROOT/RAttrBorder.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadExtent.hxx>

namespace ROOT {
namespace Experimental {


/** \class ROOT::Experimental::RPave
\ingroup GrafROOT7
\brief Base class for paves with text, statistic, legends, placed relative to RFrame position and adjustable height
\author Sergey Linev <s.linev@gsi.de>
\date 2020-06-18
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RPave : public RDrawable {

protected:

   RPave(const char *csstype) : RDrawable(csstype) {}

public:

   enum ECorner {
      kTopLeft = 1,
      kTopRight = 2,
      kBottomLeft = 3,
      kBottomRight = 4
   };

   RAttrBorder border{this, "border"};                     ///<! border attributes
   RAttrFill fill{this, "fill"};                           ///<! fill attributes
   RAttrText text{this, "text"};                           ///<! text attributes
   RAttrValue<RPadLength> width{this, "width", 0.4};       ///<! pave width
   RAttrValue<RPadLength> height{this, "height", 0.2};     ///<! pave height
   RAttrValue<bool> onFrame{this, "onFrame", true};        ///<! is pave assigned to frame (true) or to pad corner (false)
   RAttrValue<ECorner> corner{this, "corner", kTopRight};  ///<! frame/pad corner to which pave is bound
   RAttrValue<RPadLength> offsetX{this, "offsetX", 0.02};  ///<! offset X relative to selected frame or pad corner
   RAttrValue<RPadLength> offsetY{this, "offsetY", 0.02};  ///<! offset Y relative to selected frame or pad corner

   RPave() : RPave("pave") {}

};

} // namespace Experimental
} // namespace ROOT

#endif
