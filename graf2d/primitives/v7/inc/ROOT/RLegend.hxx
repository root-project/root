/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RLegend
#define ROOT7_RLegend

#include <ROOT/RBox.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class RLegend
\ingroup GrafROOT7
\brief A legend for several drawables
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RLegend : public RBox {

public:

   RLegend() : RBox("legend") {}

   RLegend(const RPadPos& p1, const RPadPos& p2) : RLegend() { SetP1(p1); SetP2(p2); }
};

} // namespace Experimental
} // namespace ROOT

#endif
