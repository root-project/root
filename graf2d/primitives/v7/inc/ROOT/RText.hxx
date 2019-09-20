/// \file ROOT/RText.hxx
/// \ingroup Graf ROOT7
/// \author Olivier Couet <Olivier.Couet@cern.ch>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
 A text.
 */

class RText : public RDrawable {

   std::string fText;                      ///< text to display
   RPadPos fPos;                           ///< position
   RAttrValues fAttr{"text"};              ///< attributes
   RAttrText  fTextAttr{fAttr, "text_"};   ///<! text attributes

public:

   RText() = default;

   RText(const std::string &txt) : RText() { fText = txt; }

   RText(const RPadPos& p, const std::string &txt) : RText() { fText = txt; fPos = p; }

   RText &SetText(const std::string &t) { fText = t; return *this; }
   const std::string &GetText() const { return fText; }

   RText &SetPos(const RPadPos& p) { fPos = p; return *this; }
   const RPadPos &GetPos() const { return fPos; }

   RAttrText &AttrText() { return fTextAttr; }
   const RAttrText &AttrText() const { return fTextAttr; }
};


} // namespace Experimental
} // namespace ROOT

#endif
