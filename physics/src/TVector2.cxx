// @(#)root/physics:$Name:  $:$Id: TVector2.cxx,v 1.1.1.1 2000/05/16 17:00:45 rdm Exp $
// Author: Pasha Murat   12/02/99
//------------------------------------------------------------------------------
// Copyright(c) 1995-1997, P.Murat (CDF collaboration, FNAL)
//
// Permission to use, copy, modify and distribute this software and its
// documentation for non-commercial purposes is hereby granted without fee,
// provided that the above copyright notice appears in all copies and
// that both the copyright notice and this permission notice appear in
// the supporting documentation. The authors make no claims about the
// suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//------------------------------------------------------------------------------

#include "TVector2.h"
#include "TClass.h"

ClassImp(TVector2)

//______________________________________________________________________________
TVector2::TVector2()
{
  fX = 0.;
  fY = 0.;
}

//______________________________________________________________________________
TVector2::TVector2(Double_t *v)
{
  fX = v[0];
  fY = v[1];
}

//______________________________________________________________________________
TVector2::TVector2(Double_t x0, Double_t y0)
{
  fX = x0;
  fY = y0;
}

//______________________________________________________________________________
TVector2::~TVector2()
{
}

//______________________________________________________________________________
TVector2 TVector2::Rotate (Double_t phi)
{
  return TVector2( fX*TMath::Cos(phi)-fY*TMath::Sin(phi), fX*TMath::Sin(phi)+fY*TMath::Cos(phi) );
}


//______________________________________________________________________________
void TVector2::Streamer(TBuffer &R__b)
{
   // Stream an object of class TVector2.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TVector2::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      if (R__v < 2) TObject::Streamer(R__b);
      R__b >> fX;
      R__b >> fY;
      R__b.CheckByteCount(R__s, R__c, TVector2::IsA());
      //====end of old versions
      
   } else {
      TVector2::Class()->WriteBuffer(R__b,this);
   }
}
