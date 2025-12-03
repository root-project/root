/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualGeoChecker
#define ROOT_TVirtualGeoChecker

#include "TObject.h"

class TGeoVolume;
class TGeoShape;
class TGeoNode;
class TGeoManager;
class TH2F;
class TStopwatch;

class TVirtualGeoChecker : public TObject {
protected:
   static TVirtualGeoChecker *fgGeoChecker; // Pointer to checker instance

public:
   TVirtualGeoChecker();
   ~TVirtualGeoChecker() override;

   virtual void
   CheckPoint(Double_t x = 0, Double_t y = 0, Double_t z = 0, Option_t *option = "", Double_t safety = 0.) = 0;
   virtual void CheckShape(TGeoShape *shape, Int_t testNo, Int_t nsamples, Option_t *option) = 0;
   virtual void CheckBoundaryErrors(Int_t ntracks = 1000000, Double_t radius = -1.) = 0;
   virtual void CheckBoundaryReference(Int_t icheck = -1) = 0;
   virtual void CheckGeometryFull(Bool_t checkoverlaps = kTRUE, Bool_t checkcrossings = kTRUE, Int_t nrays = 10000,
                                  const Double_t *vertex = nullptr) = 0;
   virtual void CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const = 0;
   virtual void CheckOverlaps(const TGeoVolume *vol, Double_t ovlp = 0.1, Option_t *option = "") = 0;
   virtual TH2F *LegoPlot(Int_t ntheta = 60, Double_t themin = 0., Double_t themax = 180., Int_t nphi = 90,
                          Double_t phimin = 0., Double_t phimax = 360., Double_t rmin = 0., Double_t rmax = 9999999,
                          Option_t *option = "") = 0;
   virtual void PrintOverlaps() const = 0;
   virtual void RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option) = 0;
   virtual void RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz,
                           const char *target_vol = nullptr, Bool_t check_norm = kFALSE) = 0;
   virtual void OpProgress(const char *opname, Long64_t current, Long64_t size, TStopwatch *watch = nullptr,
                           Bool_t last = kFALSE, Bool_t refresh = kFALSE, const char *msg = "") = 0;
   virtual TGeoNode *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char *g3path) = 0;
   virtual void SetSelectedNode(TGeoNode *node) = 0;
   virtual void SetNmeshPoints(Int_t npoints = 1000) = 0;
   virtual void Test(Int_t npoints, Option_t *option) = 0;
   virtual void TestOverlaps(const char *path) = 0;
   virtual Bool_t TestVoxels(TGeoVolume *vol, Int_t npoints = 1000000) = 0;
   virtual Double_t Weight(Double_t precision = 0.01, Option_t *option = "v") = 0;

   TVirtualGeoChecker *GeoChecker();

   ClassDefOverride(TVirtualGeoChecker, 0) // Abstract interface for geometry painters
};

#endif
