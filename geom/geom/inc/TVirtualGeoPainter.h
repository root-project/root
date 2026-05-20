// @(#)root/geom:$Id$
// Author: Andrei Gheata   11/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualGeoPainter
#define ROOT_TVirtualGeoPainter

#include "TObject.h"

class TGeoVolume;
class TGeoNode;
class TGeoShape;
class TGeoMatrix;
class TGeoHMatrix;
class TGeoManager;
class TVirtualGeoTrack;
class TParticle;
class TObjArray;
class TGeoBatemanSol;
class TGeoIteratorPlugin;
class TGeoPolygon;

class TVirtualGeoPainter : public TObject {

protected:
   static TVirtualGeoPainter *fgGeoPainter; // Pointer to class painter

public:
   enum EGeoVisLevel { kGeoVisLevel = 0 };
   enum EGeoVisOption {
      kGeoVisDefault = 0, // default visualization - everything visible 3 levels down
      kGeoVisLeaves = 1,  // only last leaves are visible
      kGeoVisOnly = 2,    // only current volume is drawn
      kGeoVisBranch = 3,  // only a given branch is drawn
      kGeoVisChanged = 4  // visibility changed
   };
   enum EGeoBombOption {
      kGeoNoBomb = 0,  // default - no bomb
      kGeoBombXYZ = 1, // explode view in cartesian coordinates
      kGeoBombCyl = 2, // explode view in cylindrical coordinates (R, Z)
      kGeoBombSph = 3  // explode view in spherical coordinates (R)
   };

public:
   TVirtualGeoPainter(TGeoManager *manager);
   ~TVirtualGeoPainter() override;

   virtual void AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys) = 0;
   virtual TVirtualGeoTrack *AddTrack(Int_t id, Int_t pdgcode, TObject *particle) = 0;
   virtual void AddTrackPoint(Double_t *point, Double_t *box, Bool_t reset = kFALSE) = 0;
   virtual void BombTranslation(const Double_t *tr, Double_t *bombtr) = 0;
   virtual Int_t CountVisibleNodes() = 0;
   virtual void DefaultAngles() = 0;
   virtual void DefaultColors() = 0;
   virtual Int_t DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py) = 0;
   virtual void DrawBatemanSol(TGeoBatemanSol *sol, Option_t *option = "") = 0;
   virtual void DrawShape(TGeoShape *shape, Option_t *option = "") = 0;
   virtual void DrawOnly(Option_t *option = "") = 0;
   virtual void DrawOverlap(void *ovlp, Option_t *option = "") = 0;
   virtual void DrawCurrentPoint(Int_t color) = 0;
   virtual void DrawPanel() = 0;
   virtual void DrawPath(const char *path, Option_t *option = "") = 0;
   virtual void DrawPolygon(const TGeoPolygon *poly) = 0;
   virtual void DrawVolume(TGeoVolume *vol, Option_t *option = "") = 0;
   virtual void EditGeometry(Option_t *option = "") = 0;
   virtual void EstimateCameraMove(Double_t /*tmin*/, Double_t /*tmax*/, Double_t *, Double_t *) {}
   virtual void ExecuteShapeEvent(TGeoShape *shape, Int_t event, Int_t px, Int_t py) = 0;
   virtual void ExecuteManagerEvent(TGeoManager *geom, Int_t event, Int_t px, Int_t py) = 0;
   virtual void ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t px, Int_t py) = 0;
   virtual Int_t GetColor(Int_t base, Float_t light) const = 0;
   virtual Int_t GetNsegments() const = 0;
   virtual void GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const = 0;
   virtual Int_t GetBombMode() const = 0;
   virtual const char *GetDrawPath() const = 0;
   virtual TGeoVolume *GetDrawnVolume() const = 0;
   virtual TGeoVolume *GetTopVolume() const = 0;
   virtual void GetViewAngles(Double_t & /*longitude*/, Double_t & /*latitude*/, Double_t & /*psi*/) {}
   virtual Int_t GetVisLevel() const = 0;
   virtual Int_t GetVisOption() const = 0;
   virtual const char *GetVolumeInfo(const TGeoVolume *volume, Int_t px, Int_t py) const = 0;
   virtual void GrabFocus(Int_t nfr = 0, Double_t dlong = 0, Double_t dlat = 0, Double_t dpsi = 0) = 0;
   virtual Double_t *GetViewBox() = 0;
   virtual Bool_t IsPaintingShape() const = 0;
   virtual Bool_t IsRaytracing() const = 0;
   virtual Bool_t IsExplodedView() const = 0;
   virtual void ModifiedPad(Bool_t update = kFALSE) const = 0;
   void Paint(Option_t *option = "") override = 0;
   virtual void PaintNode(TGeoNode *node, Option_t *option = "", TGeoMatrix *global = nullptr) = 0;
   virtual void PaintShape(TGeoShape *shape, Option_t *option = "") = 0;
   virtual void PaintOverlap(void *ovlp, Option_t *option = "") = 0;
   virtual void PaintVolume(TGeoVolume *vol, Option_t *option = "", TGeoMatrix *global = nullptr) = 0;
   virtual void Raytrace(Option_t *option = "") = 0;
   virtual void
   SetBombFactors(Double_t bombx = 1.3, Double_t bomby = 1.3, Double_t bombz = 1.3, Double_t bombr = 1.3) = 0;
   virtual void SetClippingShape(TGeoShape *shape) = 0;
   virtual void SetExplodedView(Int_t iopt = 0) = 0;
   virtual void SetGeoManager(TGeoManager *geom) = 0;
   virtual void SetIteratorPlugin(TGeoIteratorPlugin *plugin) = 0;
   virtual void SetNsegments(Int_t nseg = 20) = 0;
   virtual void SetRaytracing(Bool_t flag = kTRUE) = 0;
   static TVirtualGeoPainter *GeoPainter();
   static void SetPainter(const TVirtualGeoPainter *painter);
   virtual void SetTopVisible(Bool_t vis = kTRUE) = 0;
   virtual void SetTopVolume(TGeoVolume *vol) = 0;
   virtual void SetVisLevel(Int_t level = 3) = 0;
   virtual void SetVisOption(Int_t option = 0) = 0;
   virtual Int_t ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const = 0;
   virtual void UnbombTranslation(const Double_t *tr, Double_t *bombtr) = 0;

   ClassDefOverride(TVirtualGeoPainter, 0) // Abstract interface for geometry painters
};

#endif
