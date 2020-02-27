// @(#)root/eve7:$Id$
// Author: Sergey Linev, 27.02.2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveGeoPainter
#define ROOT7_REveGeoPainter

#include "TVirtualGeoPainter.h"

namespace ROOT {
namespace Experimental {

class REveGeoPainter : public TVirtualGeoPainter {

public:
   REveGeoPainter(TGeoManager *manager);
   virtual ~REveGeoPainter();

   void       AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys) override {}
   TVirtualGeoTrack *AddTrack(Int_t id, Int_t pdgcode, TObject *particle) override { return nullptr; }
   void       AddTrackPoint(Double_t *point, Double_t *box, Bool_t reset=kFALSE) override {}
   virtual void       BombTranslation(const Double_t *tr, Double_t *bombtr) override {}
   virtual void       CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="") override {}
   virtual void       CheckShape(TGeoShape *shape, Int_t testNo, Int_t nsamples, Option_t *option) override {}
   virtual void       CheckBoundaryErrors(Int_t ntracks=1000000, Double_t radius=-1.) override {}
   virtual void       CheckBoundaryReference(Int_t icheck=-1) override {}
   virtual void       CheckGeometryFull(Bool_t checkoverlaps=kTRUE, Bool_t checkcrossings=kTRUE, Int_t nrays=10000, const Double_t *vertex=NULL) override {}
   virtual void       CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const override {}
   virtual void       CheckOverlaps(const TGeoVolume *vol, Double_t ovlp=0.1, Option_t *option="") const override {}
   virtual Int_t      CountVisibleNodes() override { return 0; }
   virtual void       DefaultAngles() override {}
   virtual void       DefaultColors() override {}
   virtual Int_t      DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py) override { return 0; }
   virtual void       Draw(Option_t *option="") override {}
   virtual void       DrawBatemanSol(TGeoBatemanSol *sol, Option_t *option="") override {}
   virtual void       DrawShape(TGeoShape *shape, Option_t *option="") override {}
   virtual void       DrawOnly(Option_t *option="") override {}
   virtual void       DrawOverlap(void *ovlp, Option_t *option="") override {}
   virtual void       DrawCurrentPoint(Int_t color) override {}
   virtual void       DrawPanel() override {}
   virtual void       DrawPath(const char *path, Option_t *option="") override {}
   virtual void       DrawPolygon(const TGeoPolygon *poly) override {}
   virtual void       DrawVolume(TGeoVolume *vol, Option_t *option="") override {}
   virtual void       EditGeometry(Option_t *option="") override {}
   virtual void       EstimateCameraMove(Double_t /*tmin*/, Double_t /*tmax*/, Double_t *, Double_t * ) override {}
   virtual void       ExecuteShapeEvent(TGeoShape *shape, Int_t event, Int_t px, Int_t py) override {}
   virtual void       ExecuteManagerEvent(TGeoManager *geom, Int_t event, Int_t px, Int_t py) override {}
   virtual void       ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t px, Int_t py) override {}
   virtual Int_t      GetColor(Int_t base, Float_t light) const override { return 0; }
   virtual Int_t      GetNsegments() const override { return 1; }
   virtual void       GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const override {}
   virtual Int_t      GetBombMode() const override { return 0; }
   virtual const char *GetDrawPath() const override { return ""; }
   virtual TGeoVolume *GetDrawnVolume() const override { return nullptr; }
   virtual TGeoVolume *GetTopVolume() const override { return nullptr; }
   virtual void       GetViewAngles(Double_t &/*longitude*/, Double_t &/*latitude*/, Double_t &/*psi*/) override {}
   virtual Int_t      GetVisLevel() const override { return 0; }
   virtual Int_t      GetVisOption() const override { return 0; }
   virtual const char*GetVolumeInfo(const TGeoVolume *volume, Int_t px, Int_t py) const override { return "info"; }
   virtual void       GrabFocus(Int_t nfr=0, Double_t dlong=0, Double_t dlat=0, Double_t dpsi=0) override {}
   virtual Double_t  *GetViewBox() override { return nullptr; }
   virtual Bool_t     IsPaintingShape() const override { return kFALSE; }
   virtual Bool_t     IsRaytracing() const override { return kFALSE; }
   virtual Bool_t     IsExplodedView() const override { return kFALSE; }
   virtual TH2F      *LegoPlot(Int_t ntheta=60, Double_t themin=0., Double_t themax=180.,
                            Int_t nphi=90, Double_t phimin=0., Double_t phimax=360.,
                            Double_t rmin=0., Double_t rmax=9999999, Option_t *option="") override { return nullptr; }
   virtual void       ModifiedPad(Bool_t update=kFALSE) const override {}
   virtual void       OpProgress(const char *opname, Long64_t current, Long64_t size, TStopwatch *watch=0, Bool_t last=kFALSE, Bool_t refresh=kFALSE, const char *msg="") override {}
   virtual void       Paint(Option_t *option="") override {}
   virtual void       PaintNode(TGeoNode *node, Option_t *option="", TGeoMatrix* global=0) override {}
   virtual void       PaintShape(TGeoShape *shape, Option_t *option="") override {}
   virtual void       PaintOverlap(void *ovlp, Option_t *option="") override {}
   virtual void       PrintOverlaps() const override {}
   virtual void       PaintVolume(TGeoVolume *vol, Option_t *option="", TGeoMatrix* global=0) override {}
   virtual void       RandomPoints(const TGeoVolume *vol, Int_t npoints, Option_t *option="") override {}
   virtual void       RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz, const char *target_vol, Bool_t check_norm) override {}
   virtual void       Raytrace(Option_t *option="") override {}
   virtual TGeoNode  *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path) override { return nullptr; }
   virtual void       SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3,
                                     Double_t bombr=1.3) override {}
   virtual void       SetClippingShape(TGeoShape *shape) override {}
   virtual void       SetExplodedView(Int_t iopt=0) override {}
   virtual void       SetGeoManager(TGeoManager *geom) override {}
   virtual void       SetIteratorPlugin(TGeoIteratorPlugin *plugin) override {}
   virtual void       SetCheckedNode(TGeoNode *node) override {}
   virtual void       SetNsegments(Int_t nseg=20) override {}
   virtual void       SetNmeshPoints(Int_t npoints) override {}
   virtual void       SetRaytracing(Bool_t flag=kTRUE) override {}
   virtual void       SetTopVisible(Bool_t vis=kTRUE) override {}
   virtual void       SetTopVolume(TGeoVolume *vol) override {}
   virtual void       SetVisLevel(Int_t level=3) override {}
   virtual void       SetVisOption(Int_t option=0) override {}
   virtual Int_t      ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const override { return 0; }
   virtual void       Test(Int_t npoints, Option_t *option) override {}
   virtual void       TestOverlaps(const char *path) override {}
   virtual Bool_t     TestVoxels(TGeoVolume *vol) override { return kFALSE; }
   virtual void       UnbombTranslation(const Double_t *tr, Double_t *bombtr) override {}
   virtual Double_t   Weight(Double_t precision, Option_t *option="v") override { return 0.; }

   ClassDefOverride(REveGeoPainter,0)  // Web-based geo painter
};

} // namespace Experimental
} // namespace ROOT


#endif
