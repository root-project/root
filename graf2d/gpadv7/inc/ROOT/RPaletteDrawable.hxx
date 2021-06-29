/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPaletteDrawable
#define ROOT7_RPaletteDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPalette.hxx>

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RPaletteDrawable
\ingroup GrafROOT7
\brief A color palette draw near the frame.
\author Sergey Linev <s.linev@gsi.de>
\date 2020-03-05
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RPaletteDrawable final : public RDrawable {

   RPalette                fPalette;                              ///<  color palette to draw

protected:

   bool IsFrameRequired() const final { return true; }

   RPaletteDrawable() : RDrawable("palette") {}

public:

   RAttrValue<bool>        visible{this, "visible", true};       ///<! visibility flag
   RAttrValue<RPadLength>  margin{this, "margin", 0.02_normal};  ///<! horizontal margin to frame
   RAttrValue<RPadLength>  width{this, "width", 0.05_normal};    ///<! width of palette

   RPaletteDrawable(const RPalette &palette) : RPaletteDrawable() { fPalette = palette; }
   RPaletteDrawable(const RPalette &palette, bool _visible) : RPaletteDrawable() { fPalette = palette; visible = _visible; }
   const RPalette &GetPalette() const { return fPalette; }
};

//inline auto GetDrawable(const RPalette &palette)
//{
//   return std::make_shared<RPaletteDrawable>(palette);
//}


} // namespace Experimental
} // namespace ROOT

#endif
