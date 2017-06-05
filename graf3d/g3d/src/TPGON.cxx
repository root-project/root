// @(#)root/g3d:$Id$
// Author: Nenad Buncic   29/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPGON.h"
#include "TMath.h"

ClassImp(TPGON);

/** \class TPGON
\ingroup g3d
A polygon.

\image html g3d_pgon.png

It has the following parameters:

  - name:       name of the shape
  - title:      shape's title
  - material:   (see TMaterial)
  - phi1:       the azimuthal angle &phi at which the volume begins (angles
                are counted counterclockwise)
  - dphi1:      opening angle of the volume, which extends from phi1 to
                phi1+dphi
  - npdv:       number of sides of the cross section between the given
                phi limits
  - nz:         number of planes perpendicular to the z axis where
                the dimension of the section is given -- this number
                should be at least 2
  - rmin:       array of dimension nz with minimum radius at a given plane
  - rmax:       array of dimension nz with maximum radius at a given plane
  - z:          array of dimension nz with z position of given plane
*/

////////////////////////////////////////////////////////////////////////////////
/// PGON shape default constructor.

TPGON::TPGON ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// PGON shape normal constructor.
///
///  Parameters of the nz positions must be entered via TPCON::DefineSection.

TPGON::TPGON (const char *name, const char *title, const char *material, Float_t phi1,
              Float_t dphi1, Int_t npdv, Int_t nz)
     : TPCON (name, title,material, phi1, dphi1, nz)
{
   SetNumberOfDivisions (npdv);
}

////////////////////////////////////////////////////////////////////////////////
/// PGON shape default destructor.

TPGON::~TPGON ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the table of cos and sin to prepare drawing

void TPGON::FillTableOfCoSin(Double_t phi, Double_t angstep,Int_t n) const
{
   Double_t factor = 1./TMath::Cos(angstep/2);
   Double_t ph = phi-angstep;
   for (Int_t j = 0; j < n; j++) {
      ph += angstep;
      fCoTab[j] = factor*TMath::Cos(ph);
      fSiTab[j] = factor*TMath::Sin(ph);
   }
}
