// @(#)root/g3d:$Name:  $:$Id: TUtil3D.cxx,v 1.4 2002/04/11 11:41:31 rdm Exp $
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
// 3-D view utility functions                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TUtil3D.h"
#include "TROOT.h"
#include "TList.h"
#include "TAxis3D.h"
#include "TPolyLine3D.h"

ClassImp(TUtil3D)

//______________________________________________________________________________
TUtil3D::TUtil3D()
{
   SetName("R__TVirtualUtil3D");
   TUtil3D *u = (TUtil3D*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtil3D");
   if (!u) gROOT->GetListOfSpecials()->Add(this);
}

//______________________________________________________________________________
TUtil3D::~TUtil3D()
{
}

//______________________________________________________________________________
void TUtil3D::DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax)
{
   TPolyLine3D::DrawOutlineCube(outline,rmin,rmax);
}

//______________________________________________________________________________
void TUtil3D::ToggleRulers(TVirtualPad *pad)
{
   TAxis3D::ToggleRulers(pad);
}

//______________________________________________________________________________
void TUtil3D::ToggleZoom(TVirtualPad *pad)
{
   TAxis3D::ToggleZoom(pad);
}
