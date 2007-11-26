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

#include <TEveUtil.h>

#include <TObject.h>

class TEveTrans;

class TEveGridStepper : public TObject
{
private:
   Int_t *ls[3], *ns[3]; //! Internal traversal variables.

   TEveGridStepper(const TEveGridStepper&);            // Not implemented
   TEveGridStepper& operator=(const TEveGridStepper&); // Not implemented

public:
   enum StepMode_e { SM_XYZ, SM_YXZ, SM_XZY };
   StepMode_e Mode;      // Stepping mode, order of filling.

   Int_t   nx, ny, nz;   // Current positions during filling / traversal.
   Int_t   Nx, Ny, Nz;   // Number of slots in eaxh direction.
   Float_t Dx, Dy, Dz;   // Step size in each direction.
   Float_t Ox, Oy, Oz;   // Initial offset for each direction.

   TEveGridStepper(Int_t sm=SM_XYZ);
   virtual ~TEveGridStepper() {}

   void Reset();
   void Subtract(TEveGridStepper& s);
   void SetNs(Int_t nx, Int_t ny, Int_t nz=1)
   { Nx = nx; Ny = ny; Nz = nz; }
   void SetDs(Float_t dx, Float_t dy, Float_t dz=0)
   { Dx = dx; Dy = dy; Dz = dz; }
   void SetOs(Float_t ox, Float_t oy, Float_t oz=0)
   { Ox = ox; Oy = oy; Oz = oz; }

   Bool_t Step();

   void GetPosition(Float_t* p);

   void SetTrans(TEveTrans* mx);
   void SetTransAdvance(TEveTrans* mx);

   ClassDef(TEveGridStepper, 1); // Provide discrete position coordinates for placement of objects on regular grids.
}; // end class TEveGridStepper

#endif
