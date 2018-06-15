/// \file ROOT/RMarker.hxx
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

#ifndef ROOT7_RMarker
#define ROOT7_RMarker

#include <ROOT/RDrawable.hxx>
#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RDrawingOptsBase.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadPainter.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RMarker
 A simple marker.
 */

class RMarker : public RDrawableBase<RMarker> {
public:

/** class ROOT::Experimental::RMarker::DrawingOpts
 Drawing options for RMarker.
 */

class DrawingOpts: public RDrawingOptsBase {
   RDrawingAttr<RColor> fColor{*this, "Marker.Color", RColor::kBlack}; ///< The marker color.
   RDrawingAttr<float> fSize{*this, "Marker.Size", 1.};                ///< The marker size.
   RDrawingAttr<int> fStyle{*this, "Marker.Style", 1};                 ///< The marker style.
   RDrawingAttr<float> fOpacity{*this, "Marker.Opacity", 1.};          ///< The marker opacity.

public:
   /// The color of the marker.
   void SetMarkerColor(const RColor &col) { fColor = col; }
   RDrawingAttr<RColor> &GetMarkerColor() { return fColor; }
   const RColor &GetMarkerColor() const   { return fColor.Get(); }

   ///The size of the marker.
   void SetMarkerSize(float size) { fSize = size; }
   RDrawingAttr<float> &GetMarkerSize() { return fSize; }
   int GetMarkerSize() const   { return (float)fSize; }

   ///The style of the marker.
   void SetMarkerStyle(int style) { fStyle = style; }
   RDrawingAttr<int> &GetMarkerStyle() { return fStyle; }
   int GetMarkerStyle() const { return (int)fStyle; }

   ///The opacity of the marker.
   void SetMarkerColorAlpha(float opacity) { fOpacity = opacity; }
   RDrawingAttr<float> &GetMarkerColorAlpha() { return fOpacity; }
   float GetMarkerColorAlpha() const { return (float)fOpacity; }
};


private:

   /// Marker's position
   RPadPos fP;

   /// Marker's attributes
   DrawingOpts fOpts;

public:

   RMarker() = default;

   RMarker(const RPadPos& p) : fP(p) {}

   void SetP(const RPadPos& p) { fP = p; }

   const RPadPos& GetP() const { return fP; }

   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::RPadPainter &topPad) final
   {
      topPad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::ROrdinaryDisplayItem<ROOT::Experimental::RMarker>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::RMarker>
GetDrawable(const std::shared_ptr<ROOT::Experimental::RMarker> &marker)
{
   /// A RMarker is a RDrawable itself.
   return marker;
}

} // namespace Experimental
} // namespace ROOT

#endif
