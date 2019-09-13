/// \file ROOT/RLine.hxx
/// \ingroup Graf ROOT7
/// \author Olivier Couet <Olivier.Couet@cern.ch>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
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
#include <ROOT/RPadPainter.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

class RLine : public RDrawable {

   RDrawableAttributes fAttr{"line"};       ///< attributes
   RPadPos fP1{fAttr, "p1_"};               ///<! line begin
   RPadPos fP2{fAttr, "p2_"};               ///<! line end
   RAttrLine  fLineAttr{fAttr, "line_"};    ///<! line attributes

public:

   RLine() = default;

   RLine(const RPadPos& p1, const RPadPos& p2) : RLine()
   {
      fP1 = p1;
      fP2 = p2;
   }

   void SetP1(const RPadPos& p1) { fP1 = p1; }
   void SetP2(const RPadPos& p2) { fP2 = p2; }

   const RPadPos& GetP1() const { return fP1; }
   const RPadPos& GetP2() const { return fP2; }

   RAttrLine &AttrLine() { return fLineAttr; }
   const RAttrLine &AttrLine() const { return fLineAttr; }
};

} // namespace Experimental
} // namespace ROOT

#endif
