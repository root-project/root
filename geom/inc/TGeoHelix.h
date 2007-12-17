// @(#)root/geom:$Id$
// Author: Andrei Gheata   28/04/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoHelix
#define ROOT_TGeoHelix


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoHMatrix;

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoHelix - A helix defined by:                                        //
//     x = (1/c) * COS(phi)                                               //
//     y = (1/c) * SIN(phi)                                               //
//     z = s * alfa                                                       //
// where:                                                                 //
//     c = 1/Rxy  - curvature in XY plane                                 //
//     phi        - phi angle                                             //
//     S = 2*PI*s - vertical separation between helix loops               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoHelix  : public TObject
{
private :
   Double_t           fC;              // curvature in XY plane
   Double_t           fS;              // Z step of the helix / 2*PI
   Double_t           fStep;           // current step
   Double_t           fPhi;            // phi angle
   Double_t           fPointInit[3];   // initial point
   Double_t           fDirInit[3];     // normalized initial direction
   Double_t           fPoint[3];       // point after a step
   Double_t           fDir[3];         // direction after a step
   Double_t           fB[3];           // normalized direction for magnetic field
   Int_t              fQ;              // right/left-handed (+/- 1) - "charge"
   TGeoHMatrix       *fMatrix;         // transformation of local helix frame to MARS   
 
public:
   enum EGeoHelixTypes {
      kHelixNeedUpdate =   BIT(16),
      kHelixStraigth   =   BIT(17),
      kHelixCircle     =   BIT(18)
   };
   // constructors
   TGeoHelix();
   TGeoHelix(Double_t curvature, Double_t step, Int_t charge=1);
   // destructor
   virtual ~TGeoHelix();

   void            InitPoint(Double_t x0, Double_t y0, Double_t z0);
   void            InitPoint(Double_t *point);
   void            InitDirection(Double_t dirx, Double_t diry, Double_t dirz, Bool_t is_normalized=kTRUE);
   void            InitDirection(Double_t *dir, Bool_t is_normalized=kTRUE);
   
   Double_t        ComputeSafeStep(Double_t epsil=1E-6) const;
   const Double_t *GetCurrentPoint() const {return fPoint;}
   const Double_t *GetCurrentDirection() const {return fDir;}
   Double_t        GetXYcurvature() const {return fC;}
   Double_t        GetStep() const {return fStep;}
   Double_t        GetTotalCurvature() const;
   Bool_t          IsRightHanded() const {return (fQ>0)?kFALSE:kTRUE;} // a positive charge in B field makes a left-handed helix

   void            ResetStep();
   Double_t        StepToPlane(Double_t *point, Double_t *norm);
//   Double_t       *StepToPlane(Double_t a, Double_t b, Double_t c);
   
   void            SetCharge(Int_t charge);
   void            SetXYcurvature(Double_t curvature);
   void            SetField(Double_t bx, Double_t by, Double_t bz, Bool_t is_normalized=kTRUE);
   void            SetHelixStep(Double_t hstep);
   
   void            Step(Double_t step);
   
   void            UpdateHelix();

   ClassDef(TGeoHelix, 1)              // helix class
};

#endif

