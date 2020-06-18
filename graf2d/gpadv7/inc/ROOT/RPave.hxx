/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPave
#define ROOT7_RPave

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {


/** \class ROOT::Experimental::RPave
\ingroup GrafROOT7
\brief Base class for paves with text, statistic, legends, placed relative to RFrame position and adjustable height
\author Sergey Linev <s.linev@gsi.de>
\date 2020-06-18
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RPave : public RDrawable {

   class RPaveAttrs : public RAttrBase {
      friend class RPave;
      R__ATTR_CLASS(RPaveAttrs, "", AddPadLength("cornerx",0.02).AddPadLength("cornery",0.02).AddPadLength("width",0.5).AddPadLength("height",0.2));
   };

   RAttrText fAttrText{this, "text_"};       ///<! text attributes
   RAttrLine fAttrBorder{this, "border_"};   ///<! border attributes
   RAttrFill fAttrFill{this, "fill_"};       ///<! line attributes
   RPaveAttrs fAttr{this, ""};               ///<! pave direct attributes

protected:

   bool IsFrameRequired() const final { return true; }

public:

   RPave() : RDrawable("pave") {}

   RPave &SetCornerX(const RPadLength &pos)
   {
      fAttr.SetValue("cornerx", pos);
      return *this;
   }

   RPadLength GetCornerX() const
   {
      return fAttr.template GetValue<RPadLength>("cornerx");
   }

   RPave &SetCornerY(const RPadLength &pos)
   {
      fAttr.SetValue("cornery", pos);
      return *this;
   }

   RPadLength GetCornerY() const
   {
      return fAttr.template GetValue<RPadLength>("cornery");
   }

   RPave &SetWidth(const RPadLength &width)
   {
      fAttr.SetValue("width", width);
      return *this;
   }

   RPadLength GetWidth() const
   {
      return fAttr.template GetValue<RPadLength>("width");
   }

   RPave &SetHeight(const RPadLength &height)
   {
      fAttr.SetValue("height", height);
      return *this;
   }

   RPadLength GetHeight() const
   {
      return fAttr.template GetValue<RPadLength>("height");
   }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RPave &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrLine &GetAttrBorder() const { return fAttrBorder; }
   RPave &SetAttrBorder(const RAttrLine &border) { fAttrBorder = border; return *this; }
   RAttrLine &AttrBorder() { return fAttrBorder; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   RPave &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }
};

} // namespace Experimental
} // namespace ROOT

#endif
