// @(#)root/geom:$Name:  $:$Id: TGeoChecker.h,v 1.7 2003/02/07 13:46:48 brun Exp $
// Author: Andrei Gheata   01/11/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoChecker
#define ROOT_TGeoChecker

#ifndef ROOT_TObject
#include "TObject.h"
#endif

// forward declarations
class TTree;
class TGeoVolume;
class TGeoVoxelFinder;
class TGeoNode;
class TGeoManager;
class TH2F;

/*************************************************************************
 * TGeoChecker - A simple checker generating random points inside a 
 *   geometry. Generates a tree of points on the surfaces coresponding
 *   to the safety of each generated point
 *
 *************************************************************************/

class TGeoChecker : public TObject
{
private :
// data members
   TGeoManager     *fGeom;            // pointer to geometry manager
   TTree           *fTreePts;         // tree of points
   TGeoVolume      *fVsafe;           // volume to which a safety sphere node was added
// methods

public:
   // constructors
   TGeoChecker();
   TGeoChecker(TGeoManager *geom);
   TGeoChecker(const char *treename, const char *filename);
   // destructor
   virtual ~TGeoChecker();
   // methods
   void             CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const;
   void             CheckOverlaps(const TGeoVolume *vol, Double_t ovlp=0.1, Option_t *option="") const;
   void             CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="");
   Double_t         CheckVoxels(TGeoVolume *vol, TGeoVoxelFinder *voxels, Double_t *xyz, Int_t npoints);
   void             CreateTree(const char *treename, const char *filename);
   void             Generate(UInt_t npoints=1000000);      // compute safety and fill the tree
   TH2F            *LegoPlot(Int_t ntheta=60, Double_t themin=0., Double_t themax=180.,
                            Int_t nphi=90, Double_t phimin=0., Double_t phimax=360.,
                            Double_t rmin=0., Double_t rmax=9999999, Option_t *option="");
   void             Raytrace(Double_t *startpoint, UInt_t npoints=1000000);
   void             RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option);
   void             RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz);
   TGeoNode        *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path);
   void             ShootRay(Double_t *start, Double_t dirx, Double_t diry, Double_t dirz, Double_t *array, Int_t &nelem, Int_t &dim, Double_t *enpoint=0) const;
   void             ShowPoints(Option_t *option="");
   void             Test(Int_t npoints, Option_t *option);
   void             TestOverlaps(const char *path);
   Bool_t           TestVoxels(TGeoVolume *vol, Int_t npoints=1000000);
   
  ClassDef(TGeoChecker, 1)               // a simple geometry checker
};

#endif

