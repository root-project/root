/// \file ROOT/RBox.hxx
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

#ifndef ROOT7_RBox
#define ROOT7_RBox

#include <ROOT/RAttrBox.hxx>
#include <ROOT/RDrawable.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadPainter.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RBox
 A simple box.
 */

class RBox : public RDrawableBase<RBox> {
public:

/** class ROOT::Experimental::RBox::DrawingOpts
 Drawing options for RBox.
 */

class DrawingOpts: public RDrawingOptsBase, public RAttrBox {
public:
      DrawingOpts():
         RAttrBox(FromOption, "box", *this)
      {}

      using RAttrBox::RAttrBox;
};


private:

   /// Box's coordinates

   RPadPos fP1;           ///< 1st point, bottom left
   RPadPos fP2;           ///< 2nd point, top right

   /// Box's attributes
   DrawingOpts fOpts;

public:

   RBox() = default;

   RBox(const RPadPos& p1, const RPadPos& p2) : fP1(p1), fP2(p2) {}

   void SetP1(const RPadPos& p1) { fP1 = p1; }
   void SetP2(const RPadPos& p2) { fP2 = p2; }

   const RPadPos& GetP1() const { return fP1; }
   const RPadPos& GetP2() const { return fP2; }

   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::RPadPainter &topPad) final
   {
      topPad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RBox>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::RBox>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RBox> &box)
{
   /// A RBox is a RDrawable itself.
   return box;
}

} // namespace Experimental
} // namespace ROOT

#endif
