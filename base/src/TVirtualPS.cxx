// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   05/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPS.h"

TVirtualPS *gVirtualPS = 0;

ClassImp(TVirtualPS)

//______________________________________________________________________________
//
//  TVirtualPS is an abstract interface to a Postscript driver
//

//______________________________________________________________________________
TVirtualPS::TVirtualPS()
{
//*-*-*-*-*-*-*-*-*-*-*VirtualPS default constructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================
}

//______________________________________________________________________________
TVirtualPS::TVirtualPS(const char *name, Int_t)
          : TNamed(name,"Postscript interface")
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*VirtualPS constructor*-*-*-*-*-*-*-*-*-*-*-*-*

}

//______________________________________________________________________________
TVirtualPS::~TVirtualPS()
{
//*-*-*-*-*-*-*-*-*-*-*VirtualPS destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================
}

