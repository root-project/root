// @(#)root/geom:$Name:  $:$Id: TVirtualGeoPainter.h,v 1.3 2002/07/15 15:32:25 brun Exp $
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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGeoPainter                                                   //
//                                                                      //
// Abstract base class for geometry painters                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoVolume;
class TGeoNode;
class TGeoShape;
class TGeoManager;
class TObjArray;

class TVirtualGeoPainter : public TObject {


protected:
   static TVirtualGeoPainter   *fgGeoPainter; //Pointer to class painter

public:
enum EGeoVisLevel {
   kGeoVisLevel   = 0
};         
enum EGeoVisOption {
   kGeoVisDefault = 0,    // default visualization - everything visible 3 levels down
   kGeoVisLeaves  = 1,    // only last leaves are visible
   kGeoVisOnly    = 2,    // only current volume is drawn
   kGeoVisBranch  = 3     // only a given branch is drawn
};
enum EGeoBombOption {   
   kGeoNoBomb     = 0,    // default - no bomb
   kGeoBombXYZ    = 1,    // explode view in cartesian coordinates
   kGeoBombCyl    = 2,    // explode view in cylindrical coordinates (R, Z)
   kGeoBombSph    = 3     // explode view in spherical coordinates (R)
};

public:
   TVirtualGeoPainter();
   virtual ~TVirtualGeoPainter();

   virtual void       AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys) = 0;
   virtual void       BombTranslation(const Double_t *tr, Double_t *bombtr) = 0;
   virtual void       CheckPoint(Double_t x=0, Double_t y=0, Double_t z=0, Option_t *option="") = 0;
   virtual void       DefaultAngles() = 0;
   virtual void       DefaultColors() = 0;
   virtual Int_t      DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py) = 0;
   virtual void       Draw(Option_t *option="") = 0;
   virtual void       DrawOnly(Option_t *option="") = 0;
   virtual void       DrawCurrentPoint(Int_t color) = 0;
   virtual void       DrawPanel() = 0;
   virtual void       DrawPath(const char *path) = 0;
   virtual void       ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t px, Int_t py) = 0;
   virtual Int_t      GetNsegments() const = 0; 
   virtual void       GetBombFactors(Double_t &bombx, Double_t &bomby, Double_t &bombz, Double_t &bombr) const = 0;
   virtual Int_t      GetBombMode() const = 0; 
   virtual const char *GetDrawPath() const = 0; 
   virtual Int_t      GetVisLevel() const = 0; 
   virtual Int_t      GetVisOption() const = 0; 
   virtual char      *GetVolumeInfo(TGeoVolume *volume, Int_t px, Int_t py) const = 0;
   virtual Bool_t     IsExplodedView() const = 0;
   virtual Bool_t     IsOnScreen(const TGeoNode *node) const = 0;
   virtual void       ModifiedPad() const = 0;
   virtual void       Paint(Option_t *option="") = 0;
   virtual void       PaintBox(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintTube(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintTubs(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintSphere(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintPcon(TGeoVolume *vol, Option_t *option="") = 0;
   virtual void       PaintNode(TGeoNode *node, Option_t *option="") = 0;
   virtual void       RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option="") = 0;
   virtual void       RandomRays(Int_t nrays) = 0;
   virtual TGeoNode  *SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil, const char* g3path) = 0;
   virtual void       SetBombFactors(Double_t bombx=1.3, Double_t bomby=1.3, Double_t bombz=1.3,
                                     Double_t bombr=1.3) = 0;
   virtual void       SetExplodedView(UInt_t iopt=0) = 0;
   virtual void       SetNsegments(Int_t nseg) = 0;    
   static  TVirtualGeoPainter *GeoPainter();
   static void        SetPainter(const TVirtualGeoPainter *painter);
   virtual void       SetVisLevel(Int_t level=3) = 0;
   virtual void       SetVisOption(Int_t option=0) = 0;
   virtual void       Sizeof3D(const TGeoVolume *vol) const = 0;      
   virtual Int_t      ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const = 0;
   virtual void       Test(Int_t npoints, Option_t *option) = 0;
   virtual void       TestOverlaps(const char *path) = 0;
   virtual void       UnbombTranslation(const Double_t *tr, Double_t *bombtr) = 0;
      
  ClassDef(TVirtualGeoPainter,0)  //Abstract interface for geometry painters
};

#endif
