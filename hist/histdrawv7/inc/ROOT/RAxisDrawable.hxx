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
#include <ROOT/RPadLength.hxx>

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
   RPadPos fPos;                          ///< axis start point
   RPadLength fLength;                    ///< axis size
   bool fVertical{false};                 ///< is vertical axis
   RAttrAxis fAttrAxis{this, "axis_"};    ///<! axis attributes

public:

   RAxisDrawable() : RDrawable("axis") {}

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

   RAxisDrawable &SetVertical(bool vertical = true) { fVertical = vertical; return *this; }
   RAxisDrawable &SetPos(const RPadPos& pos) { fPos = pos; return *this; }
   RAxisDrawable &SetLength(const RPadLength& len) { fLength = len; return *this; }

   bool IsVertical() const { return fVertical; }
   const RPadPos& GetPos() const { return fPos; }
   const RPadLength& GetLength() const { return fLength; }

   const RAttrAxis &GetAttrAxis() const { return fAttrAxis; }
   RAxisDrawable &SetAttrAxis(const RAttrAxis &attr) { fAttrAxis = attr; return *this; }
   RAttrAxis &AttrAxis() { return fAttrAxis; }
};

} // namespace Experimental
} // namespace ROOT

#endif
