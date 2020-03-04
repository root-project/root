/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFrameTitle
#define ROOT7_RFrameTitle

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RPadPos.hxx>

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RFrameTitle
\ingroup GrafROOT7
\brief A title for the RFrame.
\author Sergey Linev <s.linev@gsi.de>
\date 2020-02-26
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RFrameTitle final : public RDrawable {

   class RTitleAttrs : public RAttrBase {
      friend class RFrameTitle;
      R__ATTR_CLASS(RTitleAttrs, "", AddString("margin","0.02").AddString("height","0.05"));
   };

   std::string fText;                    ///< title to display
   RAttrText  fAttrText{this, "text_"};  ///<! text attributes
   RTitleAttrs fAttr{this,""};           ///<! title direct attributes

protected:

   bool IsFrameRequired() const final { return true; }

public:

   RFrameTitle() : RDrawable("title") {}

   RFrameTitle(const std::string &txt) : RFrameTitle() { fText = txt; }

   RFrameTitle &SetText(const std::string &t) { fText = t; return *this; }
   const std::string &GetText() const { return fText; }

   RFrameTitle &SetMargin(const RPadLength &pos)
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

   RFrameTitle &SetHeight(const RPadLength &pos)
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
   RFrameTitle &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }
};

} // namespace Experimental
} // namespace ROOT

#endif
