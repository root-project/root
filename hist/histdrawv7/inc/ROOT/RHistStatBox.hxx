/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistStatBox
#define ROOT7_RHistStatBox

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RPadPos.hxx>

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RHistStatBox
\ingroup GrafROOT7
\brief Statistic box for RHist class
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-01
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RHistStatBox final : public RDrawable {

   class RHistStatBoxAttrs : public RAttrBase {
      friend class RHistStatBox;
      R__ATTR_CLASS(RHistStatBoxAttrs, "", AddString("cornerx","0.02").AddString("cornery","0.02").AddString("width","0.5").AddString("height","0.2"));
   };

   RAttrText  fAttrText{this, "text_"};      ///<! text attributes
   RAttrLine fAttrBorder{this, "border_"};   ///<! border attributes
   RAttrFill fAttrFill{this, "fill_"};       ///<! line attributes
   RHistStatBoxAttrs fAttr{this,""};         ///<! title direct attributes

protected:

   bool IsFrameRequired() const final { return true; }

public:

   RHistStatBox() : RDrawable("stats") {}

   RHistStatBox &SetCornerX(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("cornerx");
      else
         fAttr.SetValue("cornerx", pos.AsString());

      return *this;
   }

   RPadLength GetCornerX() const
   {
      RPadLength res;
      auto value = fAttr.GetValue<std::string>("cornerx");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RHistStatBox &SetCornerY(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("cornery");
      else
         fAttr.SetValue("cornery", pos.AsString());

      return *this;
   }

   RPadLength GetCornerY() const
   {
      RPadLength res;
      auto value = fAttr.GetValue<std::string>("cornery");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RHistStatBox &SetWidth(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("width");
      else
         fAttr.SetValue("width", pos.AsString());

      return *this;
   }

   RPadLength GetWidth() const
   {
      RPadLength res;
      auto value = fAttr.GetValue<std::string>("width");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RHistStatBox &SetHeight(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("height");
      else
         fAttr.SetValue("height", pos.AsString());

      return *this;
   }

   RPadLength GetHeight() const
   {
      RPadLength res;
      auto value = fAttr.GetValue<std::string>("height");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RHistStatBox &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrLine &GetAttrBorder() const { return fAttrBorder; }
   RHistStatBox &SetAttrBorder(const RAttrLine &border) { fAttrBorder = border; return *this; }
   RAttrLine &AttrBorder() { return fAttrBorder; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   RHistStatBox &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }
};

} // namespace Experimental
} // namespace ROOT

#endif
