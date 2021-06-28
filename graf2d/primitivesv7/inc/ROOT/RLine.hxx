/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RLine
#define ROOT7_RLine

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** \class RLine
\ingroup GrafROOT7
\brief A simple line.
\author Olivier Couet <Olivier.Couet@cern.ch>
\author Sergey Linev <S.Linev@gsi.de>
\date 2017-10-16
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RLine : public RDrawable {

   RPadPos fP1, fP2;                                   ///< line begin/end
   RAttrValue<bool> fOnFrame{this, "onframe", false};  ///<! is drawn on the frame or not
   RAttrValue<bool> fClipping{this, "clipping", false}; ///<! is clipping on when drawn on the frame
public:

   RAttrLine line{this, "line"};                  ///<! line attributes

   RLine() : RDrawable("line") {}

   RLine(const RPadPos &p1, const RPadPos &p2) : RLine()
   {
      fP1 = p1;
      fP2 = p2;
   }

   RLine &SetP1(const RPadPos &p1) { fP1 = p1; return *this; }
   RLine &SetP2(const RPadPos &p2) { fP2 = p2; return *this; }

   const RPadPos &GetP1() const { return fP1; }
   const RPadPos &GetP2() const { return fP2; }

   RLine &SetOnFrame(bool on = true) { fOnFrame = on; return *this; }
   bool GetOnFrame() const { return fOnFrame; }

   RLine &SetClipping(bool on = true) { fClipping = on; return *this; }
   bool GetClipping() const { return fClipping; }
};

} // namespace Experimental
} // namespace ROOT

#endif
