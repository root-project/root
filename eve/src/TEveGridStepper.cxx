// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveGridStepper.h>
#include <TEveTrans.h>

//______________________________________________________________________________
// TEveGridStepper
//
// Provide discrete position coordinates for placement of objects on
// regular grids.

ClassImp(TEveGridStepper)

//______________________________________________________________________________
TEveGridStepper::TEveGridStepper(Int_t sm) :
   Mode(StepMode_e(sm)),
   nx(0), ny(0), nz(0), Nx(0), Ny(0), Nz(0),
   Dx(0), Dy(0), Dz(0), Ox(0), Oy(0), Oz(0)
{
   switch(Mode) {
      default:
      case SM_XYZ:
         ls[0] = &Nx; ls[1] = &Ny; ls[2] = &Nz;
         ns[0] = &nx; ns[1] = &ny; ns[2] = &nz;
         break;
      case SM_YXZ:
         ls[0] = &Ny; ls[1] = &Nx; ls[2] = &Nz;
         ns[0] = &ny; ns[1] = &nx; ns[2] = &nz;
         break;
      case SM_XZY:
         ls[0] = &Nx; ls[1] = &Nz; ls[2] = &Ny;
         ns[0] = &nx; ns[1] = &nz; ns[2] = &ny;
         break;
   }

   nx = ny = nz = 0;
   Nx = Ny = Nz = 16;
   Dx = Dy = Dz = 1;
   Ox = Oy = Oz = 0;
}

//______________________________________________________________________________
void TEveGridStepper::Reset()
{
   nx = ny = nz = 0;
}

//______________________________________________________________________________
void TEveGridStepper::Subtract(TEveGridStepper& s)
{
   Ox = -(s.Ox + s.nx*s.Dx);
   Oy = -(s.Oy + s.ny*s.Dy);
   Oz = -(s.Oz + s.nz*s.Dz);
}
/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveGridStepper::Step()
{
   (*ns[0])++;
   if (*ns[0] >= *ls[0]) {
      *ns[0] = 0; (*ns[1])++;
      if (*ns[1] >= *ls[1]) {
         *ns[1] = 0; (*ns[2])++;
         if (*ns[2] >= *ls[2]) {
            return kFALSE;
         }
      }
   }
   return kTRUE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGridStepper::GetPosition(Float_t* p)
{
   p[0] = Ox + nx*Dx; p[1] = Oy + ny*Dy; p[2] = Oz + nz*Dz;
}

//______________________________________________________________________________
void TEveGridStepper::SetTrans(TEveTrans* mx)
{
   mx->SetPos(Ox + nx*Dx, Oy + ny*Dy, Oz + nz*Dz);
}

//______________________________________________________________________________
void TEveGridStepper::SetTransAdvance(TEveTrans* mx)
{
   SetTrans(mx);
   Step();
}
