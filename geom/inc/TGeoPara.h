// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPara
#define ROOT_TGeoPara

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoPara - parallelipeped class. It has 6 parameters :                 //
//         dx, dy, dz - half lengths in X, Y, Z                           //
//         alpha - angle w.r.t the Y axis from center of low Y edge to    //
//                 center of high Y edge [deg]                            //
//         theta, phi - polar and azimuthal angles of the segment between //
//                 low and high Z surfaces [deg]                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

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
   TGeoPara(const char *name, Double_t dx, Double_t dy, Double_t dz, Double_t alpha, Double_t theta, Double_t phi);
   TGeoPara(Double_t *param);
   // destructor
   virtual ~TGeoPara();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual Int_t         GetByteCount() const {return 48;}
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual Int_t         GetNmeshVertices() const {return 8;}
   Double_t              GetX() const  {return fX;}
   Double_t              GetY() const  {return fY;}
   Double_t              GetZ() const  {return fZ;}
   Double_t              GetAlpha() const {return fAlpha;}
   Double_t              GetTheta() const {return fTheta;}
   Double_t              GetPhi() const   {return fPhi;}
   Double_t              GetTxy() const {return fTxy;}
   Double_t              GetTxz() const {return fTxz;}
   Double_t              GetTyz() const {return fTyz;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoPara, 1)         // box primitive
};

#endif
