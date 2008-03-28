// @(#)root/mathcore:$Id$
// Author: Rene Brun, Lorenzo Moneta  17/05/2006

//////////////////////////////////////////////////////////////////////////
//
// TRandom2
//
// Random number generator class based on the maximally quidistributed combined
// Tausworthe generator by L'Ecuyer.
//
// The period of the generator is 2**88 (about 10**26) and it uses only 3 words
// for the state.
//
// For more information see:
// P. L'Ecuyer, Mathematics of Computation, 65, 213 (1996)
// P. L'Ecuyer, Mathematics of Computation, 68, 225 (1999)
//
// The publication are available online at
//  http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme.ps
//  http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme2.ps
//////////////////////////////////////////////////////////////////////////

#include "TRandom2.h"
#include "TRandom3.h"


ClassImp(TRandom2)

//______________________________________________________________________________
TRandom2::TRandom2(UInt_t seed)
{
//*-*-*-*-*-*-*-*-*-*-*default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===================

   SetName("Random2");
   SetTitle("Random number generator with period of about  10**26");
   SetSeed(seed);
}

//______________________________________________________________________________
TRandom2::~TRandom2()
{
//*-*-*-*-*-*-*-*-*-*-*default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==================

}

//______________________________________________________________________________
Double_t TRandom2::Rndm(Int_t)
{
   //  TausWorth generator from L'Ecuyer, uses as seed 3x32bits integers
   //  Use a mask of 0xffffffffUL to make in work on 64 bit machines
   //  Periodicity of about  10**26

#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d) & 0xffffffffUL ) ^ ((((s <<a) & 0xffffffffUL )^s) >>b)

   const double kScale = 2.3283064365386963e-10;    // range in 32 bit ( 1/(2**32)

   fSeed  = TAUSWORTHE (fSeed, 13, 19, 4294967294UL, 12);
   fSeed1 = TAUSWORTHE (fSeed1, 2, 25, 4294967288UL, 4);
   fSeed2 = TAUSWORTHE (fSeed2, 3, 11, 4294967280UL, 17);

   UInt_t iy = fSeed ^ fSeed1 ^ fSeed2;
   if (iy) return  kScale*static_cast<Double_t>(iy);
   return Rndm();
}

//______________________________________________________________________________
void TRandom2::RndmArray(Int_t n, Float_t *array)
{
   // Return an array of n random numbers uniformly distributed in ]0,1]

   const double kScale = 2.3283064365386963e-10;    // range in 32 bit ( 1/(2**32)

   UInt_t iy;

   for(Int_t i=0; i<n; i++) {
      fSeed  = TAUSWORTHE (fSeed, 13, 19, 4294967294UL, 12);
      fSeed1 = TAUSWORTHE (fSeed1, 2, 25, 4294967288UL, 4);
      fSeed2 = TAUSWORTHE (fSeed2, 3, 11, 4294967280UL, 17);

      iy = fSeed ^ fSeed1 ^ fSeed2;
      if (iy) array[i] = (Float_t)(kScale*static_cast<Double_t>(iy));
      else    array[i] = Rndm();
   }
}

//______________________________________________________________________________
void TRandom2::RndmArray(Int_t n, Double_t *array)
{
   // Return an array of n random numbers uniformly distributed in ]0,1]

   const double kScale = 2.3283064365386963e-10;    // range in 32 bit ( 1/(2**32)

   UInt_t iy;
   for(Int_t i=0; i<n; i++) {
      fSeed  = TAUSWORTHE (fSeed, 13, 19, 4294967294UL, 12);
      fSeed1 = TAUSWORTHE (fSeed1, 2, 25, 4294967288UL, 4);
      fSeed2 = TAUSWORTHE (fSeed2, 3, 11, 4294967280UL, 17);

      iy = fSeed ^ fSeed1 ^ fSeed2;
      if (iy) array[i] = kScale*static_cast<Double_t>(iy);
      else    array[i] = Rndm();
   }
}

//______________________________________________________________________________
void TRandom2::SetSeed(UInt_t seed)
{
   // Set the generator seed.
   // If the seed given is zero, generate automatically seed values which
   // are different every time by using TRandom3  and TUUID
   // If a seed is given generate the other two needed for the generator state using
   // a linear congruential generator
   // The only condition, stated at the end of the 1999 L'Ecuyer paper is that the seeds
   // must be greater than 1,7 and 15.

#define LCG(n) ((69069 * n) & 0xffffffffUL)  // linear congurential generator

   if (seed > 0) {
      fSeed = LCG (seed);
      if (fSeed < 2) fSeed += 2UL;
      fSeed1 = LCG (fSeed);
      if (fSeed1 < 8) fSeed1 += 8UL;
      fSeed2 = LCG (fSeed1);
      if (fSeed2 < 16) fSeed2 += 16UL;
   } else {
      // initialize using TRandom3 which uses a UUID
      TRandom3 r3(0);
      fSeed   = static_cast<UInt_t> (4294967296.*r3.Rndm());
      fSeed1  = static_cast<UInt_t> (4294967296.*r3.Rndm());
      fSeed2  = static_cast<UInt_t> (4294967296.*r3.Rndm());

      if (fSeed < 2)   fSeed  += 2UL;
      if (fSeed1 < 8)  fSeed1 += 8UL;
      if (fSeed2 < 16) fSeed2 += 16UL;
   }

   // "warm it up" by calling it 6 times
   for (int i = 0; i < 6; ++i)
      Rndm();

   return;
}

