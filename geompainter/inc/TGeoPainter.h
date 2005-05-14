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
class TVirtualGeoTrack;
class TGeoPhysicalNode;
class TGeoChecker;
class TGeoOverlap;
class TH2F;

class TGeoPainter : public TVirtualGeoPainter {
private:
   Double_t           fBombX;            // bomb factor on X
   Double_t           fBombY;            // bomb factor on Y
   Double_t           fBombZ;            // bomb factor on Z
   Double_t           fBombR;            // bomb factor on radius (cyl or sph)
   Double_t           fCheckedBox[6];    // bounding box of checked node
   Double_t           fMat[9];           // view rotation matrix
   Int_t              fNsegments;        // number of segments approximating circles
   Int_t              fNVisNodes;        // number of visible nodes
   Int_t              fVisLevel;         // depth for drawing
   Int_t              fVisOption;        // global visualization option
   Int_t              fExplodedView;     // type of exploding current view
   Bool_t             fVisLock;          // lock for adding visible volumes
   Bool_t             fTopVisible;       // set top volume visible
   Bool_t             fPaintingOverlaps; // lock overlaps painting
   Bool_t             fIsRaytracing;     // raytracing flag
   const char        *fVisBranch;        // drawn branch
   TGeoNode          *fCheckedNode;      // checked node
   TGeoOverlap       *fOverlap;          // current overlap
   TGeoMatrix        *fMatrix;           // current local matrix in case of overlaps
   TGeoManager       *fGeoManager;       // geometry to which applies
   TGeoChecker       *fChecker;          // geometry checker
   TGeoShape         *fClippingShape;    // clipping shape
   TGeoVolume        *fLastVolume;       // last drawn volume
   TObjArray         *fVisVolumes;       // list of visible volumes
   
   void               DefineColors() const;
   void               LocalToMasterVect(const Double_t *local, Double_t *master) const;

protected:
   virtual void       ClearVisibleVolumes();

public:
   TGeoPainter(TGeoManager *manager);
   virtual ~TGeoPainter();
   virtual void       AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys);
   virtual TVirtualGeoTrack *AddTrack(Int_t id, Int_t pdgcode, TObject *part);
   virtual void       AddTrackPoint(Double_t *point, Double_t *box, Bool_t reset=kFALSE);
   virtual void       BombTranslation(const Double_t *tr, Double_t *bombtr);
   virtual void       CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const;
   virtual void       CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="");
   virtual void       CheckOverlaps(const TGeoVolume *vol, Double_t ovlp=0.1, Option_t *option="") const;
   virtual Int_t      CountVisibleNodes();
   virtual void       DefaultAngles();
   virtual void       DefaultColors();
   virtual Int_t      DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py);
   virtual void       Draw(Option_t *option="");
   virtual void       DrawOverlap(void *ovlp, Option_t *option="");
   virtual void       DrawCurrentPoint(Int_t color);
   virtual void       DrawOnly(Option_t *option="");
   virtual void       DrawPanel();
   virtual void       DrawPath(const char *path);
   virtual void       EstimateCameraMove(Double_t tmin, Double_t tmax, Double_t *start, Double_t *end);
   virtual void       ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t px, Int_t py);
   virtual char      *GetVolumeInfo(const TGeoVolume *volume, Int_t px, Int_t py) const;
   virtual void       GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const 
                                    {bombx=fBombX; bomby=fBombY; bombz=fBombZ; bombr=fBombR;}
   virtual Int_t      GetBombMode() const      {return fExplodedView;}
   virtual TGeoNode  *GetCheckedNode() {return fCheckedNode;}
   TGeoChecker       *GetChecker();
   virtual Int_t      GetColor(Int_t base, Float_t light) const;
   virtual const char *GetDrawPath() const     {return fVisBranch;}
   virtual TGeoVolume *GetDrawnVolume() const;
   virtual Int_t      GetVisLevel() const      {return fVisLevel;}
   virtual Int_t      GetVisOption() const     {return fVisOption;}
   Int_t              GetNsegments() const     {return fNsegments;}
   virtual void       GrabFocus(Int_t nfr=0, Double_t dlong=0, Double_t dlat=0, Double_t dpsi=0);
   virtual Double_t  *GetViewBox() {return &fCheckedBox[0];}
   virtual void       GetViewAngles(Double_t &longitude, Double_t &latitude, Double_t &psi);
   virtual Bool_t     IsExplodedView() const {return ((fExplodedView==kGeoVisDefault)?kFALSE:kTRUE);}
   virtual Bool_t     IsRaytracing() const {return fIsRaytracing;}
   TH2F              *LegoPlot(Int_t ntheta=60, Double_t themin=0., Double_t themax=180.,
                            Int_t nphi=90, Double_t phimin=0., Double_t phimax=360.,
                            Double_t rmin=0., Double_t rmax=9999999, Option_t *option="");
   virtual void       ModifiedPad() const;
   virtual void       Paint(Option_t *option="");
   virtual void       PaintNode(TGeoNode *node, Option_t *option="");
           Bool_t     PaintShape(const TGeoShape & shape, Option_t * option) const;
   virtual void       PaintOverlap(void *ovlp, Option_t *option="");
   virtual void       PrintOverlaps() const;
   void               PaintPhysicalNode(TGeoPhysicalNode *node, Option_t *option="");
   virtual void       RandomPoints(const TGeoVolume *vol, Int_t npoints, Option_t *option="");
   virtual void       RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz);
   virtual void       Raytrace(Option_t *option="");
   virtual TGeoNode  *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path);
   virtual void       SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3, Double_t bombr=1.3);
   virtual void       SetClippingShape(TGeoShape *shape) {fClippingShape = shape;}
   virtual void       SetExplodedView(Int_t iopt=0);
   virtual void       SetNsegments(Int_t nseg=20);
   virtual void       SetGeoManager(TGeoManager *geom) {fGeoManager=geom;}
   virtual void       SetRaytracing(Bool_t flag=kTRUE) {fIsRaytracing = flag;}
   virtual void       SetTopVisible(Bool_t vis=kTRUE);
   virtual void       SetVisLevel(Int_t level=3);
   virtual void       SetVisOption(Int_t option=0);
   virtual Int_t      ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const;   
   virtual void       Test(Int_t npoints, Option_t *option);
   virtual void       TestOverlaps(const char *path);
   virtual Bool_t     TestVoxels(TGeoVolume *vol);
   virtual void       UnbombTranslation(const Double_t *tr, Double_t *bombtr);
   virtual Double_t   Weight(Double_t precision, Option_t *option="v");

  ClassDef(TGeoPainter,0)  //geometry painter
};

#endif
