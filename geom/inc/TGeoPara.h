/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :  Andrei Gheata  - date Thu 31 Jan 2002 01:47:40 PM CET
// TGeoPara::Contains() implemented by Mihaela Gheata

#ifndef ROOT_TGeoPara
#define ROOT_TGeoPara

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

/*************************************************************************
 * TGeoPara - parallelipeped class. It has 6 parameters :
 *         dx, dy, dz - half lengths in X, Y, Z
 *         alpha - angle w.r.t the Y axis from center of low Y edge to
 *                 center of high Y edge [deg]
 *         theta, phi - polar and azimuthal angles of the segment between
 *                 low and high Z surfaces [deg]
 *
 *************************************************************************/

class TGeoPara : public TGeoBBox
{
protected :
// data members
   Double_t              fX;        // X half-length
   Double_t              fY;        // Y half-length
   Double_t              fZ;        // Z half-length
   Double_t              fAlpha;     // angle w.r.t Y from the center of low Y to the hihg Y
   Double_t              fTheta;     // polar angle of segment between low and hi Z surfaces    
   Double_t              fPhi;       // azimuthal angle of segment between low and hi Z surfaces 
   Double_t              fTxy;       // tangent of XY section angle
   Double_t              fTxz;       // tangent of XZ section angle
   Double_t              fTyz;       // tangent of XZ section angle

// methods

public:
   // constructors
   TGeoPara();
   TGeoPara(Double_t dx, Double_t dy, Double_t dz, Double_t alpha, Double_t theta, Double_t phi);
   TGeoPara(Double_t *param);
   // destructor
   virtual ~TGeoPara();
   // methods
   virtual Int_t         GetByteCount() {return 48;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;

   virtual void          ComputeBBox();
   virtual Bool_t        Contains(Double_t *point);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);

   Double_t              GetX()  {return fX;}
   Double_t              GetY()  {return fY;}
   Double_t              GetZ()  {return fZ;}
   Double_t              GetAlpha() {return fAlpha;}
   Double_t              GetTheta() {return fTheta;}
   Double_t              GetPhi()   {return fPhi;}
   Double_t              GetTxy() {return fTxy;}
   Double_t              GetTxz() {return fTxz;}
   Double_t              GetTyz() {return fTyz;}

   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
   virtual void          Sizeof3D() const;

  ClassDef(TGeoPara, 1)         // box primitive
};

#endif
