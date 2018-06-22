/// \file ROOT/TMarker.hxx
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

#ifndef ROOT7_TMarker
#define ROOT7_TMarker

#include <ROOT/TDrawable.hxx>
#include <ROOT/TDrawingAttr.hxx>
#include <ROOT/TDrawingOptsBase.hxx>
#include <ROOT/TPadPos.hxx>
#include <ROOT/TPadPainter.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TMarker
 A simple marker.
 */

class TMarker : public TDrawableBase<TMarker> {
public:

/** class ROOT::Experimental::TMarker::DrawingOpts
 Drawing options for TMarker.
 */

class DrawingOpts: public TDrawingOptsBase {
   TDrawingAttr<TColor> fColor{*this, "Marker.Color", TColor::kBlack}; ///< The marker color.
   TDrawingAttr<float> fSize{*this, "Marker.Size", 1.};                ///< The marker size.
   TDrawingAttr<int> fStyle{*this, "Marker.Style", 1};                 ///< The marker style.
   TDrawingAttr<float> fOpacity{*this, "Marker.Opacity", 1.};          ///< The marker opacity.

public:
   /// The color of the marker.
   void SetMarkerColor(const TColor &col) { fColor = col; }
   TDrawingAttr<TColor> &GetMarkerColor() { return fColor; }
   const TColor &GetMarkerColor() const   { return fColor.Get(); }

   ///The size of the marker.
   void SetMarkerSize(float size) { fSize = size; }
   TDrawingAttr<float> &GetMarkerSize() { return fSize; }
   int GetMarkerSize() const   { return (float)fSize; }

   ///The style of the marker.
   void SetMarkerStyle(int style) { fStyle = style; }
   TDrawingAttr<int> &GetMarkerStyle() { return fStyle; }
   int GetMarkerStyle() const { return (int)fStyle; }

   ///The opacity of the marker.
   void SetMarkerColorAlpha(float opacity) { fOpacity = opacity; }
   TDrawingAttr<float> &GetMarkerColorAlpha() { return fOpacity; }
   float GetMarkerColorAlpha() const { return (float)fOpacity; }
};


private:

   /// Marker's position
   TPadPos fP;

   /// Marker's attributes
   DrawingOpts fOpts;

public:

   TMarker() = default;

   TMarker(const TPadPos& p) : fP(p) {}

   void SetP(const TPadPos& p) { fP = p; }

   const TPadPos& GetP() const { return fP; }

   /// Get the drawing options.
   DrawingOpts &GetOptions() { return fOpts; }
   const DrawingOpts &GetOptions() const { return fOpts; }

   void Paint(Internal::TPadPainter &topPad) final
   {
      topPad.AddDisplayItem(
         std::make_unique<ROOT::Experimental::TOrdinaryDisplayItem<ROOT::Experimental::TMarker>>(this));
   }
};

inline std::shared_ptr<ROOT::Experimental::TMarker>
GetDrawable(const std::shared_ptr<ROOT::Experimental::TMarker> &marker)
{
   /// A TMarker is a TDrawable itself.
   return marker;
}

} // namespace Experimental
} // namespace ROOT

#endif
