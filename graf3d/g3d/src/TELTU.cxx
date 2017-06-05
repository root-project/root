// @(#)root/g3d:$Id$
// Author: Rene Brun   26/06/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TELTU.h"
#include "TNode.h"

ClassImp(TELTU);

/** \class TELTU
\ingroup g3d
A cylinder with an elliptical section. It has three
parameters: the  ellipse  semi-axis in X, the ellipse
semi-axis in Y  and the half length in Z. The equation of
the conical curve is:

     X**2/fRx**2  +  Y**2/fRy**2  =  1

ELTU is not divisible.

  - name:       name of the shape
  - title:      shape's title
  - material:  (see TMaterial)
  - rx:         the  ellipse  semi-axis   in  X
  - ry:         the  ellipse  semi-axis   in  Y
  - dz:         half-length in z
*/

////////////////////////////////////////////////////////////////////////////////
/// ELTU shape default constructor.

TELTU::TELTU()
{

}

////////////////////////////////////////////////////////////////////////////////

TELTU::TELTU(const char *name, const char *title, const char *material, Float_t rx, Float_t ry,
             Float_t dz):TTUBE (name,title,material,0,rx,dz,rx?ry/rx:1.0)
{}

////////////////////////////////////////////////////////////////////////////////
/// ELTU shape default destructor.

TELTU::~TELTU()
{
}

