// Author: Andrei Gheata   05/03/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGeoPainter
#define ROOT_TGeoPainter

#include "X3DBuffer.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGeoPainter                                                          //
//                                                                      //
// Painter for TGeo geometries                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualGeoPainter
#include "TVirtualGeoPainter.h"
#endif

#ifndef ROOT_TGeoManager
#include "TGeoManager.h"
#endif

class TGeoChecker;

class TGeoPainter : public TVirtualGeoPainter {
private:
   Double_t           fBombX;            // bomb factor on X
   Double_t           fBombY;            // bomb factor on Y
   Double_t           fBombZ;            // bomb factor on Z
   Double_t           fBombR;            // bomb factor on radius (cyl or sph)
   Int_t              fNsegments;        // number of segments approximating circles
   Int_t              fVisLevel;         // depth for drawing
   Int_t              fVisOption;        // global visualization option
   Int_t              fExplodedView;     // type of exploding current view
   const char        *fVisBranch;        // drawn branch
   TGeoManager       *fGeom;             // geometry to which applies
   TGeoChecker       *fChecker;          // geometry checker
   
public:
   TGeoPainter();
   virtual ~TGeoPainter();
   virtual void       AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys);
   virtual void       BombTranslation(const Double_t *tr, Double_t *bombtr);
   virtual void       CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="");
   virtual void       DefaultAngles();
   virtual void       DefaultColors();
   virtual Int_t      DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py);
   virtual void       Draw(Option_t *option="");
   virtual void       DrawCurrentPoint(Int_t color);
   virtual void       DrawOnly(Option_t *option="");
   virtual void       DrawPanel();
   virtual void       DrawPath(const char *path);
   virtual void       ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t px, Int_t py);
   virtual char      *GetVolumeInfo(TGeoVolume *volume, Int_t px, Int_t py) const;
   virtual void       GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const 
                                    {bombx=fBombX; bomby=fBombY; bombz=fBombZ; bombr=fBombR;}
   virtual Int_t      GetBombMode() const      {return fExplodedView;}
   TGeoChecker       *GetChecker();
   virtual const char *GetDrawPath() const     {return fVisBranch;}
   virtual Int_t      GetVisLevel() const      {return fVisLevel;}
   virtual Int_t      GetVisOption() const     {return fVisOption;}
   Int_t              GetNsegments() const     {return fNsegments;}
   virtual Bool_t     IsExplodedView() const {return ((fExplodedView==kGeoVisDefault)?kFALSE:kTRUE);}
   virtual Bool_t     IsOnScreen(const TGeoNode *node) const;
   Bool_t             IsOnScreenLoop(const TGeoNode *node, TGeoNode *current, Int_t &level) const;
   virtual void       ModifiedPad() const;
   virtual void       Paint(Option_t *option="");
   void               PaintShape(X3DBuffer *buff, Bool_t rangeView);
   void               PaintBox(TGeoVolume *vol, Option_t *option="");
   void               PaintTube(TGeoVolume *vol, Option_t *option="");
   void               PaintTubs(TGeoVolume *vol, Option_t *option="");
   void               PaintSphere(TGeoVolume *vol, Option_t *option="");
   virtual void       PaintNode(TGeoNode *node, Option_t *option="");
   void               PaintPcon(TGeoVolume *vol, Option_t *option="");
   virtual void       RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option="");
   virtual void       RandomRays(Int_t nrays);
   virtual TGeoNode  *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path);
   virtual void       SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3, Double_t bombr=1.3);
   virtual void       SetExplodedView(UInt_t iopt=0);
   void               SetNsegments(Int_t nseg) {fNsegments=nseg;}
   virtual void       SetGeoManager(TGeoManager *geom) {fGeom=geom;}
   virtual void       SetVisLevel(Int_t level=3);
   virtual void       SetVisOption(Int_t option=0);
   virtual void       Sizeof3D(const TGeoVolume *vol) const;
   virtual Int_t      ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const;   
   virtual void       Test(Int_t npoints, Option_t *option);
   virtual void       TestOverlaps(const char *path);
   virtual void       UnbombTranslation(const Double_t *tr, Double_t *bombtr);

  ClassDef(TGeoPainter,0)  //geometry painter
};

#endif
