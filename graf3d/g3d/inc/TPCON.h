// @(#)root/g3d:$Id$
// Author: Nenad Buncic   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPCON
#define ROOT_TPCON


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TPCON                                                                  //
//                                                                        //
// PCON is a polycone. It has at least 9 parameters, the lower phi limit, //
// the range in phi, the number (at least two) of z planes where the      //
// radius is changing for each z boundary and the z coordinate, the       //
// minimum radius and the maximum radius.                                 //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TShape.h"


const Int_t kDiv = 20;               //default number of divisions


class TPCON : public TShape {
protected:
   // Internal cache
   mutable Double_t   *fSiTab;       //! Table of sin(fPhi1) .... sin(fPhil+fDphi1)
   mutable Double_t   *fCoTab;       //! Table of cos(fPhi1) .... cos(fPhil+fDphi1)

   Float_t     fPhi1;        // lower phi limit
   Float_t     fDphi1;       // range in phi
   Int_t       fNdiv;        // number of divisions
   Int_t       fNz;          // number of z segments
   Float_t    *fRmin;        //[fNz] pointer to array of inside radiuses
   Float_t    *fRmax;        //[fNz] pointer to array of outside radiuses
   Float_t    *fDz;          //[fNz] pointer to array of half lengths in z

   TPCON(const TPCON&);
   TPCON& operator=(const TPCON&);

   virtual void    MakeTableOfCoSin() const;  // Create the table of the fSiTab; fCoTab
   virtual void    FillTableOfCoSin(Double_t phi, Double_t angstep,Int_t n) const; // Fill the table of cosin
   virtual void    SetPoints(Double_t *points) const;
   virtual Bool_t  SetSegsAndPols(TBuffer3D & buffer) const;

public:
   TPCON();
   TPCON(const char *name, const char *title, const char *material, Float_t phi1, Float_t dphi1, Int_t nz);
   virtual ~TPCON();

   virtual void     DefineSection(Int_t secNum, Float_t z, Float_t rmin, Float_t rmax);
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections) const;
   virtual Int_t    GetNumberOfDivisions () const {if (fNdiv) return fNdiv; else return kDiv;}
   virtual Float_t  GetPhi1() const  {return fPhi1;}
   virtual Float_t  GetDhi1() const  {return fDphi1;}
   virtual Int_t    GetNz() const    {return fNz;}
   virtual Float_t *GetRmin() const  {return fRmin;}
   virtual Float_t *GetRmax() const  {return fRmax;}
   virtual Float_t *GetDz() const    {return fDz;}
   virtual Int_t    GetNdiv() const  {return fNdiv;}
   virtual void     SetNumberOfDivisions (Int_t p);
   virtual void     Sizeof3D() const;

   ClassDef(TPCON,2)  //PCON shape
};

#endif
