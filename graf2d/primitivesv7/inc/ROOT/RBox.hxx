/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBox
#define ROOT7_RBox

#include <ROOT/ROnFrameDrawable.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RAttrBorder.hxx>
#include <ROOT/RPadPos.hxx>

#include <initializer_list>

namespace ROOT {
namespace Experimental {

/** \class RBox
\ingroup GrafROOT7
\brief A simple box.
\author Olivier Couet <Olivier.Couet@cern.ch>
\date 2017-10-16
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RBox : public ROnFrameDrawable {

   RPadPos fP1, fP2;                                   ///< box corners coordinates

protected:
   // constructor for derived classes
   RBox(const char *csstype) : ROnFrameDrawable(csstype) {}

public:

   RAttrBorder border{this, "border"};            ///<! box border attributes
   RAttrFill fill{this, "fill"};                  ///<! box fill attributes

   RBox() : RBox("box") {}

   RBox(const RPadPos &p1, const RPadPos &p2) : RBox()
   {
      fP1 = p1;
      fP2 = p2;
   }

   RBox &SetP1(const RPadPos &p1) { fP1 = p1; return *this; }
   RBox &SetP2(const RPadPos &p2) { fP2 = p2; return *this; }

   const RPadPos &GetP1() const { return fP1; }
   const RPadPos &GetP2() const { return fP2; }
};

} // namespace Experimental
} // namespace ROOT

#endif
