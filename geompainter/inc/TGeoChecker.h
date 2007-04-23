// @(#)root/geom:$Name:  $:$Id: TGeoChecker.h,v 1.17 2006/11/03 21:22:32 brun Exp $
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
class TGeoMatrix;
class TGeoOverlap;
class TBuffer3D;
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
   TGeoManager     *fGeoManager;      // pointer to geometry manager
   TGeoVolume      *fVsafe;           // volume to which a safety sphere node was added
   TBuffer3D       *fBuff1;           // Buffer containing mesh vertices for first volume
   TBuffer3D       *fBuff2;           // Buffer containing mesh vertices for second volume
   Bool_t           fFullCheck;       // Full overlap checking
// methods
   void             CleanPoints(Double_t *points, Int_t &numPoints) const;
public:
   // constructors
   TGeoChecker();
   TGeoChecker(TGeoManager *geom);
   TGeoChecker(const char *treename, const char *filename);
   // destructor
   virtual ~TGeoChecker();
   // methods
   void             CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const;
   void             CheckOverlaps(const TGeoVolume *vol, Double_t ovlp=0.1, Option_t *option="");
   void             CheckOverlapsBySampling(TGeoVolume *vol, Double_t ovlp=0.1, Int_t npoints=1000000) const;
   void             CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="");
   Double_t         CheckVoxels(TGeoVolume *vol, TGeoVoxelFinder *voxels, Double_t *xyz, Int_t npoints);
   TH2F            *LegoPlot(Int_t ntheta=60, Double_t themin=0., Double_t themax=180.,
                            Int_t nphi=90, Double_t phimin=0., Double_t phimax=360.,
                            Double_t rmin=0., Double_t rmax=9999999, Option_t *option="");
   void             PrintOverlaps() const;
   void             RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option);
   void             RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz);
   TGeoOverlap     *MakeCheckOverlap(const char *name, TGeoVolume *vol1, TGeoVolume *vol2, TGeoMatrix *mat1, TGeoMatrix *mat2, Bool_t isovlp, Double_t ovlp);
   TGeoNode        *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path);
   void             ShootRay(Double_t *start, Double_t dirx, Double_t diry, Double_t dirz, Double_t *array, Int_t &nelem, Int_t &dim, Double_t *enpoint=0) const;
   //void             ShowPoints(Option_t *option="");
   void             Test(Int_t npoints, Option_t *option);
   void             TestOverlaps(const char *path);
   Bool_t           TestVoxels(TGeoVolume *vol, Int_t npoints=1000000);
   Double_t         Weight(Double_t precision=0.01, Option_t *option="v");
   
   ClassDef(TGeoChecker, 2)               // a simple geometry checker
};

#endif

