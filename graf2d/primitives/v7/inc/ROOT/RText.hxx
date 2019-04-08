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
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadPainter.hxx>

#include <initializer_list>
#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RText
 A text.
 */

class RText : public RDrawableBase<RText> {
public:

/** class ROOT::Experimental::RText::DrawingOpts
 Drawing options for RText.
 */

class DrawingOpts: public RDrawingOptsBase, public RAttrText {
public:
   DrawingOpts(): RAttrText("text", this) {}
};

private:

   /// The text itself
   std::string fText;

   /// Text's position
   RPadPos fP;

   /// Text's attributes
   DrawingOpts fOpts;

public:

   RText() = default;

   RText(const std::string &str) : fText(str) {}
   RText(const RPadPos& p, const std::string &str) : fText(str), fP(p) {}

   void SetText(const std::string &txt) { fText = txt; }

   std::string GetText() const { return fText; }

   void SetPosition(const RPadPos& p) {fP = p;}

   const RPadPos& GetPosition() const { return fP; }


   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::RPadPainter &pad) final
   {
      pad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RText>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::RText>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RText> &text)
{
   /// A RText is a RDrawable itself.
   return text;
}

} // namespace Experimental
} // namespace ROOT

#endif
