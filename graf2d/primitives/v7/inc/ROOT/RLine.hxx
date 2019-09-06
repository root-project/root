/// \file ROOT/RLine.hxx
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

#ifndef ROOT7_RLine
#define ROOT7_RLine

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadPainter.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RLine
 A simple line.
 */

class RLine : public RDrawableBase<RLine> {
public:

/** \class ROOT::Experimental::RLine::DrawingOpts
 Drawing options for RLine.
 */
class DrawingOpts: public RDrawingOptsBase, public RAttrLine {
public:
   DrawingOpts(): RAttrLine(FromOption, "line", *this) {}
};


private:

   /// Line's coordinates

   RPadPos fP1;           ///< 1st point
   RPadPos fP2;           ///< 2nd point

   /// Line's attributes
   DrawingOpts fOpts;

public:

   RLine() = default;

   RLine(const RPadPos& p1, const RPadPos& p2) : fP1(p1), fP2(p2) {}

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
         std::make_unique<ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RLine>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::RLine>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RLine> &line)
{
   /// A RLine is a RDrawable itself.
   return line;
}


class RLineNew : public RDrawable {

   RPadPos fP1;           ///< 1st point
   RPadPos fP2;           ///< 2nd point

   RDrawingOptsBase fOpts; ///<! only temporary here, should be removed later

   RDrawableAttributes fAttr{"line"}; ///< attributes

public:

   RLineNew() = default;

   RLineNew(const RPadPos& p1, const RPadPos& p2) : fP1(p1), fP2(p2) {}

   void SetP1(const RPadPos& p1) { fP1 = p1; }
   void SetP2(const RPadPos& p2) { fP2 = p2; }

   const RPadPos& GetP1() const { return fP1; }
   const RPadPos& GetP2() const { return fP2; }

   RAttrLineNew AttrLine() { return RAttrLineNew(fAttr); }

   const RAttrLineNew AttrLine() const { return RAttrLineNew(fAttr); }

   /** TDOD: remove it later */
   RDrawingOptsBase &GetOptionsBase() override { return fOpts; }

   void Paint(Internal::RPadPainter &topPad) final
   {
      topPad.AddDisplayItem(std::make_unique<RDrawableDisplayItem>(*this));
   }

};

inline std::shared_ptr<ROOT::Experimental::RLineNew>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RLineNew> &line)
{
   /// A RLine is a RDrawable itself.
   return line;
}


} // namespace Experimental
} // namespace ROOT

#endif
