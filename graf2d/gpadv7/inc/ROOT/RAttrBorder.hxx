/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrBorder
#define ROOT7_RAttrBorder

#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrBorder
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-08
\brief Drawing line attributes for different objects.
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrBorder : public RAttrLine {

   RAttrValue<int>     fRx{this, "rx", 0};              ///<! rounding on x coordinate, px
   RAttrValue<int>     fRy{this, "ry", 0};              ///<! rounding on y coordinate, px

   R__ATTR_CLASS(RAttrBorder, "border");

   ///The rounding on x, px
   RAttrBorder &SetRx(int rx) { fRx = rx; return *this; }
   int GetRx() const { return fRx; }

   ///The rounding on x, px
   RAttrBorder &SetRy(int ry) { fRy = ry; return *this; }
   int GetRy() const { return fRy; }
};

} // namespace Experimental
} // namespace ROOT

#endif
