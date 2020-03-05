/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPaletteDraw
#define ROOT7_RPaletteDraw

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrAxis.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPalette.hxx>

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RPaletteDraw
\ingroup GrafROOT7
\brief A color palette draw near the frame.
\author Sergey Linev <s.linev@gsi.de>
\date 2020-03-05
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RPaletteDraw final : public RDrawable {

   class ROwnAttrs : public RAttrBase {
      friend class RPaletteDraw;
      R__ATTR_CLASS(ROwnAttrs, "", AddString("margin","0.02").AddString("size","0.05"));
   };

   Internal::RIOShared<RPalette> fPalette;  ///< I/O capable reference on palette
   // RPalette   fPalette;                     ///  color palette to draw
   RAttrAxis  fAttrAxis{this, "axis_"};     ///<! axis attributes
   ROwnAttrs fAttr{this,""};                ///<! own attributes

protected:

   bool IsFrameRequired() const final { return true; }

   void CollectShared(Internal::RIOSharedVector_t &vect) final { vect.emplace_back(&fPalette); }

   RPaletteDraw() : RDrawable("palette") {}

public:

   // RPaletteDraw(const RPalette &palette) : RPaletteDraw() { fPalette = palette; }
   // const RPalette &GetPalette() const { return fPalette; }

   RPaletteDraw(std::shared_ptr<RPalette> palette) : RPaletteDraw() { fPalette = palette; }

   std::shared_ptr<RPalette> GetPalette() const { return fPalette.get_shared(); }

   RPaletteDraw &SetMargin(const RPadLength &pos)
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

   RPaletteDraw &SetSize(const RPadLength &sz)
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
   RPaletteDraw &SetAttrAxis(const RAttrAxis &attr) { fAttrAxis = attr; return *this; }
   RAttrAxis &AttrAxis() { return fAttrAxis; }
};

} // namespace Experimental
} // namespace ROOT

#endif
