/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST
// TGeoShape::Contains implemented by Mihaela Gheata

#ifndef ROOT_TGeoTube
#define ROOT_TGeoTube

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

/*************************************************************************
 * TGeoTube - cylindrical tube  class. A tube has 3 parameters :
 *            Rmin - minimum radius
 *            Rmax - maximum radius 
 *            dz - half length
 *
 *************************************************************************/


class TGeoTube : public TGeoBBox
{
protected :
// data members
   Double_t              fRmin; // inner radius
   Double_t              fRmax; // outer radius
   Double_t              fDz;   // half length
// methods

public:
   // constructors
   TGeoTube();
   TGeoTube(Double_t rmin, Double_t rmax, Double_t dz);
   TGeoTube(Double_t *params);
   // destructor
   virtual ~TGeoTube();
   // methods
   virtual Int_t         GetByteCount() {return 48;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;

   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);
   static  Double_t      DistToOutS(Double_t *point, Double_t *dir, Int_t iact,Double_t step, Double_t *safe,
                                    Double_t rmin, Double_t rmax, Double_t dz);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   static  Double_t      DistToInS(Double_t *point, Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);

   virtual Double_t      GetRmin() {return fRmin;}
   virtual Double_t      GetRmax() {return fRmax;}
   virtual Double_t      GetDz()   {return fDz;}
   
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetTubeDimensions(Double_t rmin, Double_t rmax, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoTube, 1)         // cylindrical tube class

};

/*************************************************************************
 * TGeoTubeSeg - a phi segment of a tube. Has 5 parameters :
 *            - the same 3 as a tube;
 *            - first phi limit (in degrees)
 *            - second phi limit 
 *
 *************************************************************************/

class TGeoTubeSeg : public TGeoTube
{
protected:
   // data members
   Double_t              fPhi1;  // first phi limit 
   Double_t              fPhi2;  // second phi limit 

   static Double_t       DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1,
                                      Double_t s2, Double_t c2, Double_t sm, Double_t cm);   
public:
   // constructors
   TGeoTubeSeg();
   TGeoTubeSeg(Double_t rmin, Double_t rmax, Double_t dz, 
               Double_t phi1, Double_t phi2);
   TGeoTubeSeg(Double_t *params);
   // destructor
   virtual ~TGeoTubeSeg();
   // methods
   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);

   virtual Int_t         GetByteCount() {return 56;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;
   Double_t              GetPhi1() {return fPhi1;}
   Double_t              GetPhi2() {return fPhi2;}
   
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   static  Double_t      DistToOutS(Double_t *point, Double_t *dir, Int_t iact,Double_t step, Double_t *safe,
                                    Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   static  Double_t      DistToInS(Double_t *point, Double_t *dir, Double_t rmin, Double_t rmax, Double_t dz,
                                   Double_t c1, Double_t s1, Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetTubsDimensions(Double_t rmin, Double_t rmax, Double_t dz,
                                       Double_t phi1, Double_t phi2);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoTubeSeg, 1)         // cylindrical tube segment class 
};
/*************************************************************************
 * TGeoCtub - a tube segment cut with 2 planes. Has 11 parameters :
 *            - the same 5 as a tube segment;
 *            - x,y,z components of the normal to the -dZ cut plane in 
 *              point (0,0,-dZ)
 *            -  x,y,z components of the normal to the +dZ cut plane in 
 *              point (0,0,dZ)
 *
 *************************************************************************/

class TGeoCtub : public TGeoTubeSeg
{
protected:
   // data members
   Double_t             *fNlow;  // normal to lower cut plane 
   Double_t             *fNhigh; // normal to highet cut plane 
    
public:
   // constructors
   TGeoCtub();
   TGeoCtub(Double_t rmin, Double_t rmax, Double_t dz, Double_t phi1, Double_t phi2,
            Double_t lx, Double_t ly, Double_t lz, Double_t hx, Double_t hy, Double_t hz);
   TGeoCtub(Double_t *params);
   // destructor
   virtual ~TGeoCtub();
   // methods
   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);

   virtual Int_t         GetByteCount() {return 98;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;
   Double_t             *GetNlow() {return fNlow;}
   Double_t             *GetNhigh() {return fNhigh;}
   Double_t              GetZcoord(Double_t xc, Double_t yc, Double_t zc) const;
   
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);
   virtual void          InspectShape();
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   void                  SetCtubDimensions(Double_t rmin, Double_t rmax, Double_t dz,
                                       Double_t phi1, Double_t phi2, Double_t lx, Double_t ly, Double_t lz,
                                       Double_t hx, Double_t hy, Double_t hz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;

  ClassDef(TGeoCtub, 1)         // cut tube segment class 
};


#endif
