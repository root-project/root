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
#include <ROOT/RAttrOnFrame.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** \class RLine
\ingroup GrafROOT7
\brief A simple line.
\authors Olivier Couet <Olivier.Couet@cern.ch>, Sergey Linev <S.Linev@gsi.de>
\date 2017-10-16
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RLine : public RDrawable, public RAttrOnFrame, public RAttrLine {

   RPadPos fP1, fP2;                    ///< line begin/end

public:
   RLine() : RDrawable("line"), RAttrOnFrame(this), RAttrLine(this) {}

   RLine(const RPadPos &p1, const RPadPos &p2) : RLine()
   {
      fP1 = p1;
      fP2 = p2;
   }

   RLine &SetP1(const RPadPos &p1)
   {
      fP1 = p1;
      return *this;
   }

   RLine &SetP2(const RPadPos &p2)
   {
      fP2 = p2;
      return *this;
   }

   const RPadPos &GetP1() const { return fP1; }
   const RPadPos &GetP2() const { return fP2; }
};

} // namespace Experimental
} // namespace ROOT

#endif
