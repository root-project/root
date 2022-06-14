// @(#)root/g3d:$Id$
// Author: Rene Brun   08/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THYPE.h"

ClassImp(THYPE);

/** \class THYPE
\ingroup g3d
An hyperboloid (not implemented)

It has 4 parameters:

  - name:       name of the shape
  - title:      shape's title
  - material:  (see TMaterial)
  - rmin:       inner radius of the tube
  - rmax:       outer radius of the tube
  - dz:         half-length of the box along the z-axis
  - phi:        stereo angle
*/

////////////////////////////////////////////////////////////////////////////////
/// HYPE shape default constructor

THYPE::THYPE()
{
   fPhi = 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// HYPE shape normal constructor

THYPE::THYPE(const char *name, const char *title, const char *material, Float_t rmin,
             Float_t rmax, Float_t dz, Float_t phi)
      : TTUBE(name,title,material,rmin,rmax,dz)
{
   fPhi = phi;
}

////////////////////////////////////////////////////////////////////////////////
/// HYPE shape default destructor

THYPE::~THYPE()
{
}
