// @(#)root/geom:$Name:  $:$Id: TGeoChecker.h,v 1.1 2002/07/15 15:32:25 brun Exp $
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


// forward declarations
class TGeoVolume;
class TGeoNode;
class TGeoManager;
class TTree;

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
   TGeoManager     *fGeom;
   TTree           *fTreePts;
// methods

public:
   // constructors
   TGeoChecker();
   TGeoChecker(TGeoManager *geom);
   TGeoChecker(const char *treename, const char *filename);
   // destructor
   virtual ~TGeoChecker();
   // methods
   void             CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="");
   void             CreateTree(const char *treename, const char *filename);
   void             Generate(UInt_t npoints=1000000);      // compute safety and fill the tree
   void             Raytrace(Double_t *startpoint, UInt_t npoints=1000000);
   void             RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option);
   void             RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz);
   TGeoNode        *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path);
   void             ShowPoints(Option_t *option="");
   void             Test(Int_t npoints, Option_t *option);
   void             TestOverlaps(const char *path);
   
  ClassDef(TGeoChecker, 1)               // a simple geometry checker
};

#endif

