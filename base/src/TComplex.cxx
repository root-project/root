// @(#)root/base:$Name:  $:$Id: TComplex.cxx,v 1.58 2004/04/16 20:53:55 brun Exp $
// Author: Federico Carminati   22/04/2004

#include "TComplex.h"

ClassImp(TComplex)

//______________________________________________________________________________
TComplex::TComplex(Double_t re, Double_t im, Bool_t polar): 
  fRe(re),
  fIm(im)
{
  //
  // Standard constructor
  //
  if(polar) {
    fRe=re*TMath::Cos(im);
    fIm=re*TMath::Sin(im);
  }
}
