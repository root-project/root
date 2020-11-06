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
#include <ROOT/RAxis.hxx>
#include <ROOT/RLogger.hxx>
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

template<typename AxisType = RAxisEquidistant>
class RAxisDrawable : public RDrawable {

   AxisType fAxis;                        ///< axis object to draw
   RPadPos fP1;                           ///< axis begin
   RPadPos fP2;                           ///< axis end
   RAttrAxis fAttrAxis{this, "axis_"};    ///<! axis attributes

public:

   RAxisDrawable() : RDrawable("axis") {}

   RAxisDrawable(const RPadPos& p1, const RPadPos& p2) : RAxisDrawable() { fP1 = p1; fP2 = p2; }

   /*
   RAxisDrawable(const RAxisConfig &cfg) : RAxisDrawable()
   {
      switch (cfg.GetKind()) {
         case RAxisConfig::kEquidistant: fAxis = Internal::AxisConfigToType<RAxisConfig::kEquidistant>()(cfg); break;
         case RAxisConfig::kGrow: fAxis = Internal::AxisConfigToType<RAxisConfig::kGrow>()(cfg); break;
         case RAxisConfig::kIrregular: fAxis = Internal::AxisConfigToType<RAxisConfig::kIrregular>()(cfg); break;
         default: R__ERROR_HERE("HIST") << "Unhandled axis kind";
      }
   }
   */

   RAxisDrawable(const AxisType &axis) : RAxisDrawable() { fAxis = axis; }

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
