// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   04/03/99

//////////////////////////////////////////////////////////////////////////
//
// TRandom2
//
// Random number generator class (periodicity > 10**14).
//
// Has a better periodicity than its base class TRandom but is slower.
//////////////////////////////////////////////////////////////////////////

#include "TMath.h"
#include "TRandom2.h"

ClassImp(TRandom2)

//______________________________________________________________________________
TRandom2::TRandom2(UInt_t seed)
{
//*-*-*-*-*-*-*-*-*-*-*default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================

   SetName("Random2");
   SetTitle("Random number generator with period > 10**14");
   SetSeed(seed);
   fSeed1 = 9876;
   fSeed2 = 54321;
}

//______________________________________________________________________________
TRandom2::~TRandom2()
{
//*-*-*-*-*-*-*-*-*-*-*default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================

}

//______________________________________________________________________________
void TRandom2::GetSeed2(UInt_t &seed1, UInt_t &seed2)
{
//  Set the random generator seeds

   seed1 = UInt_t(fSeed1);
   seed2 = UInt_t(fSeed2);
}

//______________________________________________________________________________
Float_t TRandom2::Rndm(Int_t)
{
//  Machine independent random number generator.
//  Produces uniformly-distributed floating points between 0 and 1.
//  Identical sequence on all machines of >= 32 bits.
//  Periodicity > 10**14

   Int_t k = Int_t(fSeed1/53668);
   fSeed1 = 40014*(fSeed1 - k*53668) - k*12211;
   if (fSeed1 < 0) fSeed1 += 2147483563;
   k = Int_t(fSeed2/52774);
   fSeed2 = 40692*(fSeed2 - k*52774) - k*3791;
   if (fSeed2 < 0) fSeed2 += 2147483399;
   Double_t iz = fSeed1 - fSeed2;
   if (iz <= 0) iz += 2147483562;
   Float_t r = iz*4.6566128e-10;
   return r;
}

//______________________________________________________________________________
void TRandom2::SetSeed(UInt_t seed)
{
//  Set the random generator sequence (not yet implemented)

   fSeed = seed;
}

//______________________________________________________________________________
void TRandom2::SetSeed2(UInt_t seed1, UInt_t seed2)
{
//  Set the random generator seeds
//  Note that seed1 and seed2 must be < 2147483647

   fSeed1 = Double_t(seed1);
   fSeed2 = Double_t(seed2);
}
