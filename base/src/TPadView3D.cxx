// @(#)root/base:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   30/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPadView3D                                                           //
//                                                                      //
// TPadView3D is a generic 3D viewer.                                   //
// For a concrete viewer see TGLViewer.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPadView3D.h"
#include "TVirtualPad.h"


//   ClassImp(TPadView3D)   //3-D View of TPad

//______________________________________________________________________________
TPadView3D::~TPadView3D()
{
  // Delete 3D viewer.

  if (fParent) {
     fParent->ResetView3D();
     SetPad();
  }
}


//  Getter's / Setter's methods for the data-members

//______________________________________________________________________________
void  TPadView3D::GetRange(Double_t min[3], Double_t max[3]) const
{
  memcpy(min,fViewBoxMin,sizeof(fViewBoxMin));
  memcpy(max,fViewBoxMax,sizeof(fViewBoxMax));
}
//______________________________________________________________________________
void  TPadView3D::SetRange(Double_t min[3], Double_t max[3])
{
  memcpy(fViewBoxMin,min,sizeof(fViewBoxMin));
  memcpy(fViewBoxMax,max,sizeof(fViewBoxMax));
}

//______________________________________________________________________________
void  TPadView3D::GetShift(Double_t main_shift[3], Double_t extra_shift[3]) const
{
  memcpy(main_shift,fTranslate,sizeof(fTranslate));
  memcpy(extra_shift,fExtraTranslate,sizeof(fExtraTranslate));
}

//______________________________________________________________________________
void  TPadView3D::SetShift(Double_t main_shift[3], Double_t extra_shift[3])
{
  memcpy(fTranslate,main_shift,sizeof(fTranslate));
  memcpy(fExtraTranslate,extra_shift,sizeof(fExtraTranslate));
}

//______________________________________________________________________________
void  TPadView3D::GetAngles(Double_t main_angles[3], Double_t extra_angles[3]) const
{
  memcpy(main_angles,fAngles,sizeof(fAngles));
  memcpy(extra_angles,fExtraAngles,sizeof(fExtraAngles));
}

//______________________________________________________________________________
void  TPadView3D::SetAngles(Double_t main_angles[3], Double_t extra_angles[3])
{
  memcpy(fAngles,main_angles,sizeof(fAngles));
  memcpy(fExtraAngles,extra_angles,sizeof(fExtraAngles));
}

//______________________________________________________________________________
void  TPadView3D::GetAnglesFactors(Double_t factors[3]) const
{
  memcpy(factors,fAnglFactor,sizeof(fAnglFactor));
}
//______________________________________________________________________________
void  TPadView3D::SetAnglesFactors(Double_t factors[3])
{
  memcpy(fAnglFactor,factors,sizeof(fAnglFactor));
}

//______________________________________________________________________________
void  TPadView3D::SetScale(Float_t scale){ fScale = scale;}
