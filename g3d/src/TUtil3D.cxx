// @(#)root/g3d:$Name:  $:$Id: TUtil3D.cxx,v 1.3 2005/11/24 17:28:07 couet Exp $
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
// The functions in this class are called via the TPluginManager.       //
// see TVirtualUtil3d.h for more information .                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TUtil3D.h"
#include "TList.h"
#include "TAxis3D.h"
#include "TPolyLine3D.h"

ClassImp(TUtil3D)


//______________________________________________________________________________
TUtil3D::TUtil3D() : TVirtualUtil3D()
{
   // note that this object is automatically added to the gROOT list of specials
   // in the TVirtualUtil3D constructor.
}


//______________________________________________________________________________
TUtil3D::~TUtil3D()
{
   // TUtil3D destructor.
}


//______________________________________________________________________________
void TUtil3D::DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax)
{
   // draw the outline of a cube while rotaing a 3-d object in the pad
   
   TPolyLine3D::DrawOutlineCube(outline,rmin,rmax);
}


//______________________________________________________________________________
void TUtil3D::ToggleRulers(TVirtualPad *pad)
{
   // draw the 3 reference axis in a 3-d view in the pad
   
   TAxis3D::ToggleRulers(pad);
}


//______________________________________________________________________________
void TUtil3D::ToggleZoom(TVirtualPad *pad)
{
   // toggle zooming in a 3-d view in a pad

   TAxis3D::ToggleZoom(pad);
}
