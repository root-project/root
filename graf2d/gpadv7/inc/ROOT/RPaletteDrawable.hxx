/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPaletteDrawable
#define ROOT7_RPaletteDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrAxis.hxx>
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

   class ROwnAttrs : public RAttrBase {
      friend class RPaletteDrawable;
      R__ATTR_CLASS(ROwnAttrs, "", AddBool("visible", true).AddString("margin","0.02").AddString("size","0.05"));
   };

   RPalette   fPalette;                     ///  color palette to draw
   RAttrAxis  fAttrAxis{this, "axis_"};     ///<! axis attributes
   ROwnAttrs  fAttr{this,""};               ///<! own attributes

protected:

   bool IsFrameRequired() const final { return true; }

   RPaletteDrawable() : RDrawable("palette") {}

public:

   RPaletteDrawable(const RPalette &palette) : RPaletteDrawable() { fPalette = palette; }
   RPaletteDrawable(const RPalette &palette, bool visible) : RPaletteDrawable() { fPalette = palette; SetVisible(visible); }
   const RPalette &GetPalette() const { return fPalette; }

   RPaletteDrawable &SetVisible(bool on = true) { fAttr.SetValue("visible", on); return *this; }
   bool GetVisible() const { return fAttr.GetValue<bool>("visible"); }

   RPaletteDrawable &SetMargin(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("margin");
      else
         fAttr.SetValue("margin", pos.AsString());

      return *this;
   }

   RPadLength GetMargin() const
   {
      RPadLength res;
      auto value = fAttr.GetValue<std::string>("margin");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RPaletteDrawable &SetSize(const RPadLength &sz)
   {
      if (sz.Empty())
         fAttr.ClearValue("size");
      else
         fAttr.SetValue("size", sz.AsString());

      return *this;
   }

   RPadLength GetSize() const
   {
      RPadLength res;
      auto value = fAttr.GetValue<std::string>("size");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   const RAttrAxis &GetAttrAxis() const { return fAttrAxis; }
   RPaletteDrawable &SetAttrAxis(const RAttrAxis &attr) { fAttrAxis = attr; return *this; }
   RAttrAxis &AttrAxis() { return fAttrAxis; }
};

//inline auto GetDrawable(const RPalette &palette)
//{
//   return std::make_shared<RPaletteDrawable>(palette);
//}


} // namespace Experimental
} // namespace ROOT

#endif
