/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFrameTitle
#define ROOT7_RFrameTitle

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadPos.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

/** \class RFrameTitle
\ingroup GrafROOT7
\brief A title for the RFrame.
\author Sergey Linev <s.linev@gsi.de>
\date 2020-02-26
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/

class RFrameTitle final : public RDrawable {

   std::string fText;                                           ///< title to display

protected:

   bool IsFrameRequired() const final { return true; }

public:

   RAttrText text{this, "text", 0.07};                         ///<! title text attributes
   RAttrValue<RPadLength> margin{this, "margin", 0.02_normal}; ///<! title margin to frame
   RAttrValue<RPadLength> height{this, "height", 0.05_normal}; ///<! title height

   RFrameTitle() : RDrawable("title") {}

   RFrameTitle(const std::string &txt) : RFrameTitle() { fText = txt; }

   RFrameTitle &SetText(const std::string &t)
   {
      fText = t;
      return *this;
   }
   const std::string &GetText() const { return fText; }
};

} // namespace Experimental
} // namespace ROOT

#endif
