// @(#)root/base:$Name:  $:$Id: TRandom3.h,v 1.3 2000/11/22 08:35:44 brun Exp $
// Author: Peter Malzacher   31/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRandom3
#define ROOT_TRandom3



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandom3                                                             //
//                                                                      //
// random number generator class: Mersenne Twistor                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRandom
#include "TRandom.h"
#endif

class TRandom3 : public TRandom {

private:
   UInt_t   fMt[624];
   Int_t    fCount624;

public:
   TRandom3(UInt_t seed=65539);
   virtual ~TRandom3();
   virtual  Double_t  Rndm(Int_t i=0);
   virtual  void      RndmArray(Int_t n, Float_t *array);
   virtual  void      RndmArray(Int_t n, Double_t *array);
   virtual  void      SetSeed(UInt_t seed=0);

   ClassDef(TRandom3,2)  //Random number generator: Mersenne Twistor
};

R__EXTERN TRandom *gRandom;

#endif
