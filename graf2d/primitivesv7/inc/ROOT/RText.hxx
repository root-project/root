/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RText
#define ROOT7_RText

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RPadPos.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
\ingroup GrafROOT7
\brief A text.
\author Olivier Couet <Olivier.Couet@cern.ch>
\date 2017-10-16
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/

class RText : public RDrawable {

   std::string fText;                                  ///< text to display
   RPadPos fPos;                                       ///< position
   RAttrText fAttrText{this, "text"};                  ///<! text attributes
   RAttrValue<bool> fOnFrame{this, "onframe", false};  ///<! is drawn on the frame or not
   RAttrValue<bool> fClipping{this, "clipping", false}; ///<! is clipping on when drawn on the frame

public:
   RText() : RDrawable("text") {}

   RText(const std::string &txt) : RText() { fText = txt; }

   RText(const RPadPos &p, const std::string &txt) : RText()
   {
      fText = txt;
      fPos = p;
   }

   RText &SetText(const std::string &t)
   {
      fText = t;
      return *this;
   }
   const std::string &GetText() const { return fText; }

   RText &SetPos(const RPadPos &p)
   {
      fPos = p;
      return *this;
   }
   const RPadPos &GetPos() const { return fPos; }

   const RAttrText &AttrText() const { return fAttrText; }
   RAttrText &AttrText() { return fAttrText; }

   void SetOnFrame(bool on = true) { fOnFrame = on; }
   bool GetOnFrame() const { return fOnFrame; }

   void SetClipping(bool on = true) { fClipping = on; }
   bool GetClipping() const { return fClipping; }
};

} // namespace Experimental
} // namespace ROOT

#endif
