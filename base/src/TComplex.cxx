// @(#)root/base:$Name:  $:$Id: TComplex.cxx,v 1.1 2004/04/22 06:56:09 brun Exp $
// Author: Federico Carminati   22/04/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TComplex                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TComplex.h"

ClassImp(TComplex)

//______________________________________________________________________________
TComplex::TComplex(Double_t re, Double_t im, Bool_t polar) : fRe(re), fIm(im)
{
   // Standard constructor

   if (polar) {
      fRe=re*TMath::Cos(im);
      fIm=re*TMath::Sin(im);
   }
}
