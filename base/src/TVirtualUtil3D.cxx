// @(#)root/base:$Name:  $:$Id: TVirtualUtil3D.cxx,v 1.1 2002/09/14 16:19:14 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Abstract interface to the 3-D view utility                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualUtil3D.h"
#include "TROOT.h"

ClassImp(TVirtualUtil3D)

//______________________________________________________________________________
TVirtualUtil3D::TVirtualUtil3D()
{
   SetName("R__TVirtualUtil3D");
   TVirtualUtil3D *u = (TVirtualUtil3D*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtil3D");
   if (!u) gROOT->GetListOfSpecials()->Add(this);
}
//______________________________________________________________________________
TVirtualUtil3D::~TVirtualUtil3D()
{
}
