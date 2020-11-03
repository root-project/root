/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAxisDrawable
#define ROOT7_RAxisDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrAxis.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAxisDrawable
\ingroup GrafROOT7
\brief Drawing object for RAxis.
\author Sergey Linev <S.Linev@gsi.de>
\date 2020-11-03
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAxisDrawable : public RDrawable {

   RPadPos fP1;                            ///< axis begin
   RPadPos fP2;                            ///< axis end
   RAttrAxis fAttrAxis{this, "axis_"};    ///<! axis attributes

public:

   RAxisDrawable() : RDrawable("line") {}

   RAxisDrawable(const RPadPos& p1, const RPadPos& p2) : RAxisDrawable() { fP1 = p1; fP2 = p2; }

   RAxisDrawable &SetP1(const RPadPos& p1) { fP1 = p1; return *this; }
   RAxisDrawable &SetP2(const RPadPos& p2) { fP2 = p2; return *this; }

   const RPadPos& GetP1() const { return fP1; }
   const RPadPos& GetP2() const { return fP2; }

   const RAttrAxis &GetAttrAxis() const { return fAttrAxis; }
   RAxisDrawable &SetAttrAxis(const RAttrAxis &attr) { fAttrAxis = attr; return *this; }
   RAttrAxis &AttrAxis() { return fAttrAxis; }
};

} // namespace Experimental
} // namespace ROOT

#endif
