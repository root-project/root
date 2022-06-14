// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGridStepper
#define ROOT_TEveGridStepper

#include "TEveUtil.h"

#include "TObject.h"

class TEveTrans;

class TEveGridStepper : public TObject
{
   friend class TEveGridStepperSubEditor;

private:
   Int_t *fLimitArr[3], *fValueArr[3]; //! Internal traversal variables.

   TEveGridStepper(const TEveGridStepper&);            // Not implemented
   TEveGridStepper& operator=(const TEveGridStepper&); // Not implemented

public:
   enum EStepMode_e { kSM_XYZ, kSM_YXZ, kSM_XZY };

protected:
   EStepMode_e fMode;       // Stepping mode, order of filling.

   Int_t   fCx, fCy, fCz;   // Current positions during filling / traversal.
   Int_t   fNx, fNy, fNz;   // Number of slots in each direction.
   Float_t fDx, fDy, fDz;   // Step size in each direction.
   Float_t fOx, fOy, fOz;   // Initial offset for each direction.

public:
   TEveGridStepper(Int_t sm=kSM_XYZ);
   virtual ~TEveGridStepper() {}

   void Reset();
   void Subtract(TEveGridStepper& s);
   void SetNs(Int_t nx, Int_t ny, Int_t nz=1)
   { fNx = nx; fNy = ny; fNz = nz; }
   void SetDs(Float_t dx, Float_t dy, Float_t dz=0)
   { fDx = dx; fDy = dy; fDz = dz; }
   void SetOs(Float_t ox, Float_t oy, Float_t oz=0)
   { fOx = ox; fOy = oy; fOz = oz; }

   Bool_t Step();

   void GetPosition(Float_t* p);

   void SetTrans(TEveTrans* mx);
   void SetTransAdvance(TEveTrans* mx);

   Int_t   GetCx() const { return fCx; }
   Int_t   GetCy() const { return fCy; }
   Int_t   GetCz() const { return fCz; }
   Int_t   GetNx() const { return fNx; }
   Int_t   GetNy() const { return fNy; }
   Int_t   GetNz() const { return fNz; }
   Float_t GetDx() const { return fDx; }
   Float_t GetDy() const { return fDy; }
   Float_t GetDz() const { return fDz; }
   Float_t GetOx() const { return fOx; }
   Float_t GetOy() const { return fOy; }
   Float_t GetOz() const { return fOz; }

   ClassDef(TEveGridStepper, 1); // Provide discrete position coordinates for placement of objects on regular grids.
}; // end class TEveGridStepper

#endif
