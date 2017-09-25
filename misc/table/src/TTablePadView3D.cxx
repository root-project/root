// @(#)root/table:$Id$
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
// TTablePadView3D                                                      //
//                                                                      //
// TTablePadView3D is a generic 3D viewer.                              //
// For a concrete viewer see TGLViewer.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTablePadView3D.h"
#include "TVirtualPad.h"


//   ClassImp(TTablePadView3D);   //3-D View of TPad

////////////////////////////////////////////////////////////////////////////////
/// Delete 3D viewer.

TTablePadView3D::~TTablePadView3D()
{
   if (fParent) {
      ///  fParent->ResetView3D();
      SetPad();
   }
}


//  Getter's / Setter's methods for the data-members

////////////////////////////////////////////////////////////////////////////////
///get view range

void  TTablePadView3D::GetRange(Double_t min[3], Double_t max[3]) const
{
   memcpy(min,fViewBoxMin,sizeof(fViewBoxMin));
   memcpy(max,fViewBoxMax,sizeof(fViewBoxMax));
}
////////////////////////////////////////////////////////////////////////////////
///set view range

void  TTablePadView3D::SetRange(Double_t min[3], Double_t max[3])
{
   memcpy(fViewBoxMin,min,sizeof(fViewBoxMin));
   memcpy(fViewBoxMax,max,sizeof(fViewBoxMax));
}

////////////////////////////////////////////////////////////////////////////////
///get shift parameters

void  TTablePadView3D::GetShift(Double_t main_shift[3], Double_t extra_shift[3]) const
{
   memcpy(main_shift,fTranslate,sizeof(fTranslate));
   memcpy(extra_shift,fExtraTranslate,sizeof(fExtraTranslate));
}

////////////////////////////////////////////////////////////////////////////////
///set shift parameters

void  TTablePadView3D::SetShift(Double_t main_shift[3], Double_t extra_shift[3])
{
   memcpy(fTranslate,main_shift,sizeof(fTranslate));
   memcpy(fExtraTranslate,extra_shift,sizeof(fExtraTranslate));
}

////////////////////////////////////////////////////////////////////////////////
///get view angles

void  TTablePadView3D::GetAngles(Double_t main_angles[3], Double_t extra_angles[3]) const
{
   memcpy(main_angles,fAngles,sizeof(fAngles));
   memcpy(extra_angles,fExtraAngles,sizeof(fExtraAngles));
}

////////////////////////////////////////////////////////////////////////////////
///set view angles

void  TTablePadView3D::SetAngles(Double_t main_angles[3], Double_t extra_angles[3])
{
   memcpy(fAngles,main_angles,sizeof(fAngles));
   memcpy(fExtraAngles,extra_angles,sizeof(fExtraAngles));
}

////////////////////////////////////////////////////////////////////////////////
///get view angles factors

void  TTablePadView3D::GetAnglesFactors(Double_t factors[3]) const
{
   memcpy(factors,fAnglFactor,sizeof(fAnglFactor));
}
////////////////////////////////////////////////////////////////////////////////
///set view angles factors

void  TTablePadView3D::SetAnglesFactors(Double_t factors[3])
{
   memcpy(fAnglFactor,factors,sizeof(fAnglFactor));
}

////////////////////////////////////////////////////////////////////////////////
///set view scale

void  TTablePadView3D::SetScale(Float_t scale)
{
   fScale = scale;
}
