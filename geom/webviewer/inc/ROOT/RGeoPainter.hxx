// Author: Sergey Linev, 27.02.2020

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RGeoPainter
#define ROOT7_RGeoPainter

#include "TVirtualGeoPainter.h"

#include <ROOT/RGeomViewer.hxx>

namespace ROOT {

class RGeoPainter : public TVirtualGeoPainter {

   TGeoManager *fGeoManager{nullptr};

   std::shared_ptr<RGeomViewer> fViewer;
   Int_t fTopVisible{-1};   ///<!  is s

public:
   RGeoPainter(TGeoManager *manager);
   ~RGeoPainter() override;

   void       AddSize3D(Int_t, Int_t, Int_t) override {}
   TVirtualGeoTrack *AddTrack(Int_t, Int_t, TObject *) override;
   void       AddTrackPoint(Double_t *, Double_t *, Bool_t =kFALSE) override;
   void       BombTranslation(const Double_t *, Double_t *) override {}
   Int_t      CountVisibleNodes() override { return 0; }
   void       DefaultAngles() override {}
   void       DefaultColors() override {}
   Int_t      DistanceToPrimitiveVol(TGeoVolume *, Int_t, Int_t) override { return 0; }
   void       Draw(Option_t * ="") override {}
   void       DrawBatemanSol(TGeoBatemanSol *, Option_t * ="") override {}
   void       DrawShape(TGeoShape *, Option_t * ="") override {}
   void       DrawOnly(Option_t * ="") override {}
   void       DrawOverlap(void *, Option_t * ="") override {}
   void       DrawCurrentPoint(Int_t) override {}
   void       DrawPanel() override {}
   void       DrawPath(const char *, Option_t * ="") override {}
   void       DrawPolygon(const TGeoPolygon *) override {}
   void       DrawVolume(TGeoVolume *, Option_t * ="") override;
   void       EditGeometry(Option_t * ="") override {}
   void       EstimateCameraMove(Double_t /*tmin*/, Double_t /*tmax*/, Double_t *, Double_t * ) override {}
   void       ExecuteShapeEvent(TGeoShape *, Int_t, Int_t, Int_t) override {}
   void       ExecuteManagerEvent(TGeoManager *, Int_t, Int_t, Int_t) override {}
   void       ExecuteVolumeEvent(TGeoVolume *, Int_t, Int_t, Int_t) override {}
   Int_t      GetColor(Int_t, Float_t) const override { return 0; }
   Int_t      GetNsegments() const override { return 1; }
   void       GetBombFactors(Double_t &, Double_t &, Double_t &, Double_t &) const override {}
   Int_t      GetBombMode() const override { return 0; }
   const char *GetDrawPath() const override { return ""; }
   TGeoVolume *GetDrawnVolume() const override { return nullptr; }
   TGeoVolume *GetTopVolume() const override { return nullptr; }
   void       GetViewAngles(Double_t &/*longitude*/, Double_t &/*latitude*/, Double_t &/*psi*/) override {}
   Int_t      GetVisLevel() const override { return 0; }
   Int_t      GetVisOption() const override { return 0; }
   const char *GetVolumeInfo(const TGeoVolume *, Int_t, Int_t) const override { return "info"; }
   void       GrabFocus(Int_t =0, Double_t =0, Double_t =0, Double_t =0) override {}
   Double_t  *GetViewBox() override { return nullptr; }
   Bool_t     IsPaintingShape() const override { return kFALSE; }
   Bool_t     IsRaytracing() const override { return kFALSE; }
   Bool_t     IsExplodedView() const override { return kFALSE; }
   void       ModifiedPad(Bool_t =kFALSE) const override {}
   void       Paint(Option_t * ="") override {}
   void       PaintNode(TGeoNode *, Option_t * ="", TGeoMatrix * = nullptr) override {}
   void       PaintShape(TGeoShape *, Option_t * ="") override {}
   void       PaintOverlap(void*, Option_t * ="") override {}
   void       PaintVolume(TGeoVolume *, Option_t * = "", TGeoMatrix * = nullptr) override {}
   void       Raytrace(Option_t * = "") override {}
   void       SetBombFactors(Double_t =1.3, Double_t =1.3, Double_t =1.3, Double_t =1.3) override {}
   void       SetClippingShape(TGeoShape *) override {}
   void       SetExplodedView(Int_t =0) override {}
   void       SetGeoManager(TGeoManager *) override;
   void       SetIteratorPlugin(TGeoIteratorPlugin *) override {}
   void       SetNsegments(Int_t =20) override {}
   void       SetRaytracing(Bool_t =kTRUE) override {}
   void       SetTopVisible(Bool_t on = kTRUE) override;
   void       SetTopVolume(TGeoVolume *) override {}
   void       SetVisLevel(Int_t =3) override {}
   void       SetVisOption(Int_t =0) override {}
   Int_t      ShapeDistancetoPrimitive(const TGeoShape *, Int_t, Int_t, Int_t) const override { return 0; }
   void       UnbombTranslation(const Double_t *, Double_t *) override {}

   ClassDefOverride(RGeoPainter,0)  // Web-based geo painter
};

} // namespace ROOT


#endif
