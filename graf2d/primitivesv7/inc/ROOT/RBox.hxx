/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBox
#define ROOT7_RBox

#include <ROOT/RDrawable.hxx>
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

class RBox : public RDrawable {

   RPadPos fP1, fP2;                                   ///< box corners coordinates
   RAttrFill fAttrFill{this, "fill"};                  ///<! box fill attributes
   RAttrValue<bool> fOnFrame{this, "onframe", false};  ///<! is drawn on the frame or not
   RAttrValue<bool> fClipping{this, "clipping", false}; ///<! is clipping on when drawn on the frame

protected:
   // constructor for derived classes
   RBox(const std::string &subtype) : RDrawable(subtype) {}

public:

   RAttrBorder border{this, "border"};            ///<! box border attributes

   RBox() : RDrawable("box") {}

   RBox(const RPadPos &p1, const RPadPos &p2) : RBox()
   {
      fP1 = p1;
      fP2 = p2;
   }

   RBox &SetP1(const RPadPos &p1) { fP1 = p1; return *this; }
   RBox &SetP2(const RPadPos &p2) { fP2 = p2; return *this; }

   const RPadPos &GetP1() const { return fP1; }
   const RPadPos &GetP2() const { return fP2; }

   const RAttrFill &AttrFill() const { return fAttrFill; }
   RAttrFill &AttrFill() { return fAttrFill; }

   RBox &SetOnFrame(bool on = true) { fOnFrame = on; return *this; }
   bool GetOnFrame() const { return fOnFrame; }

   RBox &SetClipping(bool on = true) { fClipping = on; return *this; }
   bool GetClipping() const { return fClipping; }
};

} // namespace Experimental
} // namespace ROOT

#endif
