// @(#)root/g3d:$Id$
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTUBE
#define ROOT_TTUBE


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTUBE                                                                  //
//                                                                        //
// This tube has 3 parameters, the inside radius, the outside radius, and //
// the half length in z. Optional parameter is number of segments, also   //
// known as precision (default value is 20).                              //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TShape.h"


const Int_t kDivNum = 20;               //default number of divisions


class TTUBE : public TShape {
protected:

   Float_t fRmin;        // ellipse  semi-axis   in  X inside
   Float_t fRmax;        // ellipse  semi-axis   in  X outside

   Float_t fDz;          // half length in z
   Int_t   fNdiv;        // number of segments (precision)

   Float_t fAspectRatio; // defines  (the ellipse semi-axis in Y)/(the ellipse semi-axis in X)

   // Internal cache
   mutable Double_t   *fSiTab;   //! Table of sin(fPhi1) .... sin(fPhil+fDphi1)
   mutable Double_t   *fCoTab;   //! Table of cos(fPhi1) .... cos(fPhil+fDphi1)

   TTUBE(const TTUBE&);
   TTUBE& operator=(const TTUBE&);

   virtual void    MakeTableOfCoSin() const;  // Create the table of the fSiTab; fCoTab
   virtual void    SetPoints(Double_t *points) const;
   virtual void    SetSegsAndPols(TBuffer3D & buffer) const;

public:
   TTUBE();
   TTUBE(const char *name, const char *title, const char *material, Float_t rmin, Float_t rmax, Float_t dz, Float_t aspect=1);
   TTUBE(const char *name, const char *title, const char *material, Float_t rmax, Float_t dz);
   virtual ~TTUBE();

   virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections) const;
   virtual Float_t GetRmin() const  {return fRmin;}
   virtual Float_t GetRmax() const  {return fRmax;}
   virtual Float_t GetDz()   const  {return fDz;}
   virtual Int_t   GetNdiv() const  {return fNdiv;}
   virtual Float_t GetAspectRatio() const {return fAspectRatio;}
   virtual Int_t   GetNumberOfDivisions () const {if (fNdiv) return fNdiv; else return kDivNum;}
   virtual void    SetNumberOfDivisions (Int_t ndiv);
   virtual void    SetAspectRatio(Float_t factor=1){fAspectRatio = factor;}
   virtual void    Sizeof3D() const;

   ClassDef(TTUBE,3)  //TUBE shape
};

#endif
