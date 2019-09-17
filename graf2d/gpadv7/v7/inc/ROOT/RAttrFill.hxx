/// \file ROOT/RAttrLine.hxx
/// \ingroup Gpad ROOT7
/// \author Sergey Linev
/// \date 2019-09-13
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrFill
#define ROOT7_RAttrFill

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RColor.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrLine
 Drawing line attributes for different objects.
 */

class RAttrFill : public RAttrBase {

   RColor fColor{this, "color_"}; ///<! line color, will access container from line attributes

protected:
   const RAttrValues::Map_t &GetDefaults() const override
   {
      static auto dflts = RAttrValues::Map_t().AddInt("style",1).AddDefaults(fColor);
      return dflts;
   }

public:

   using RAttrBase::RAttrBase;

   RAttrFill(const RAttrFill &src) : RAttrFill() { src.CopyTo(*this); }
   RAttrFill &operator=(const RAttrFill &src) { Clear(); src.CopyTo(*this); return *this; }

   ///The fill style
   RAttrFill &SetStyle(int style) { SetValue("style", style); return *this; }
   int GetStyle() const { return GetValue<int>("style"); }

   ///The fill color
   RAttrFill &SetColor(const RColor &color) { fColor = color; return *this; }
   const RColor &Color() const { return fColor; }
   RColor &Color() { return fColor; }

};

} // namespace Experimental
} // namespace ROOT

#endif
