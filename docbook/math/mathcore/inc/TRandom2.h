// @(#)root/mathcore:$Id$
// Author: Rene Brun   04/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRandom2
#define ROOT_TRandom2



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandom2                                                             //
//                                                                      //
// random number generator class (periodicity > 10**26)                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRandom
#include "TRandom.h"
#endif

class TRandom2 : public TRandom {

protected:
   UInt_t   fSeed1;  //Random number generator seed 1
   UInt_t   fSeed2;  //Random number generator seed 2

public:
   TRandom2(UInt_t seed=1);
   virtual ~TRandom2();
   virtual  Double_t Rndm(Int_t i=0);
   virtual  void     RndmArray(Int_t n, Float_t *array);
   virtual  void     RndmArray(Int_t n, Double_t *array);
   virtual  void     SetSeed(UInt_t seed=0);

   ClassDef(TRandom2,1)  //Random number generator with periodicity of 10**26
};

R__EXTERN TRandom *gRandom;

#endif
