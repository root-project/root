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

typedef struct _x3d_points_ {
   Int_t     numPoints;
   Double_t *points;    // x0, y0, z0, x1, y1, z1, ...
} X3DPoints;   

class TGeoHMatrix;
class TGeoChecker;
class TH2F;

class TGeoPainter : public TVirtualGeoPainter {
private:
   Double_t           fBombX;            // bomb factor on X
   Double_t           fBombY;            // bomb factor on Y
   Double_t           fBombZ;            // bomb factor on Z
   Double_t           fBombR;            // bomb factor on radius (cyl or sph)
   Double_t           fCheckedBox[6];    // bounding box of checked node
   Int_t              fNsegments;        // number of segments approximating circles
   Int_t              fVisLevel;         // depth for drawing
   Int_t              fVisOption;        // global visualization option
   Int_t              fExplodedView;     // type of exploding current view
   Bool_t             fVisLock;          // lock for adding visible volumes
   Bool_t             fTopVisible;       // set top volume visible
   const char        *fVisBranch;        // drawn branch
   TGeoNode          *fCheckedNode;      // checked node
   TGeoManager       *fGeom;             // geometry to which applies
   TGeoChecker       *fChecker;          // geometry checker
   TObjArray         *fVisVolumes;       // list of visible volumes
   
public:
   TGeoPainter();
   virtual ~TGeoPainter();
   virtual void       AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys);
   virtual void       BombTranslation(const Double_t *tr, Double_t *bombtr);
   virtual void       CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const;
   virtual void       CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="");
   virtual void       CheckOverlaps(const TGeoVolume *vol, Double_t ovlp=0.1, Option_t *option="") const;
   virtual void       DefaultAngles();
   virtual void       DefaultColors();
   virtual Int_t      DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py);
   virtual void       Draw(Option_t *option="");
   virtual void       DrawCurrentPoint(Int_t color);
   virtual void       DrawOnly(Option_t *option="");
   virtual void       DrawPanel();
   virtual void       DrawPath(const char *path);
   virtual void       ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t px, Int_t py);
   virtual char      *GetVolumeInfo(const TGeoVolume *volume, Int_t px, Int_t py) const;
   virtual void       GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const 
                                    {bombx=fBombX; bomby=fBombY; bombz=fBombZ; bombr=fBombR;}
   virtual Int_t      GetBombMode() const      {return fExplodedView;}
   virtual TGeoNode  *GetCheckedNode() {return fCheckedNode;}
   TGeoChecker       *GetChecker();
   virtual const char *GetDrawPath() const     {return fVisBranch;}
   virtual Int_t      GetVisLevel() const      {return fVisLevel;}
   virtual Int_t      GetVisOption() const     {return fVisOption;}
   Int_t              GetNsegments() const     {return fNsegments;}
   virtual void       GrabFocus();
   virtual Bool_t     IsExplodedView() const {return ((fExplodedView==kGeoVisDefault)?kFALSE:kTRUE);}
   virtual Bool_t     IsOnScreen(const TGeoNode *node) const;
   TH2F              *LegoPlot(Int_t ntheta=60, Double_t themin=0., Double_t themax=180.,
                            Int_t nphi=90, Double_t phimin=0., Double_t phimax=360.,
                            Double_t rmin=0., Double_t rmax=9999999, Option_t *option="");
   virtual void       ModifiedPad() const;
   virtual void      *MakeBox3DBuffer(const TGeoVolume *vol);
   virtual void      *MakeTube3DBuffer(const TGeoVolume *vol);
   virtual void      *MakeTubs3DBuffer(const TGeoVolume *vol);
   virtual void      *MakeSphere3DBuffer(const TGeoVolume *vol);
   virtual void      *MakePcon3DBuffer(const TGeoVolume *vol);
   virtual void       Paint(Option_t *option="");
   void               PaintShape(X3DBuffer *buff, Bool_t rangeView, TGeoHMatrix *glmat);
   void               PaintBox(TGeoShape *shape, Option_t *option="", TGeoHMatrix *glmat=0);
   void               PaintCompositeShape(TGeoVolume *vol, Option_t *option="");
   void               PaintTube(TGeoShape *shape, Option_t *option="", TGeoHMatrix *glmat=0);
   void               PaintTubs(TGeoShape *shape, Option_t *option="", TGeoHMatrix *glmat=0);
   void               PaintSphere(TGeoShape *shape, Option_t *option="", TGeoHMatrix *glmat=0);
   virtual void       PaintNode(TGeoNode *node, Option_t *option="");
   void               PaintPcon(TGeoShape *shape, Option_t *option="", TGeoHMatrix *glmat=0);
   virtual void       PrintOverlaps() const;
   virtual void       RandomPoints(const TGeoVolume *vol, Int_t npoints, Option_t *option="");
   virtual void       RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz);
   virtual TGeoNode  *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path);
   virtual void       SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3, Double_t bombr=1.3);
   virtual void       SetExplodedView(Int_t iopt=0);
   virtual void       SetNsegments(Int_t nseg=20);
   virtual void       SetGeoManager(TGeoManager *geom) {fGeom=geom;}
   virtual void       SetTopVisible(Bool_t vis=kTRUE);
   virtual void       SetVisLevel(Int_t level=3);
   virtual void       SetVisOption(Int_t option=0);
   virtual void       Sizeof3D(const TGeoVolume *vol) const;
   virtual Int_t      ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const;   
   virtual void       Test(Int_t npoints, Option_t *option);
   virtual void       TestOverlaps(const char *path);
   virtual Bool_t     TestVoxels(TGeoVolume *vol);
   virtual void       UnbombTranslation(const Double_t *tr, Double_t *bombtr);

  ClassDef(TGeoPainter,0)  //geometry painter
};

#endif
