// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePlot3D.h"
#include "TEveTrans.h"


//______________________________________________________________________________
// Description of TEvePlot3D
//

ClassImp(TEvePlot3D);

//______________________________________________________________________________
TEvePlot3D::TEvePlot3D(const char* n, const char* t) :
   TEveElementList(n, t),
   fPlot(0),
   fLogX(kFALSE), fLogY(kFALSE), fLogZ(kFALSE)
{
   // Constructor.

   InitMainTrans();
}

//______________________________________________________________________________
void TEvePlot3D::Paint(Option_t* )
{
   // Paint this object. Only direct rendering is supported.

   PaintStandard(this);
}
