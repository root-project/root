// @(#)root/base:$Name:  $:$Id: TRandom2.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
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
// random number generator class (periodicity > 10**14)                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRandom
#include "TRandom.h"
#endif

class TRandom2 : public TRandom {

protected:
   Double_t   fSeed1;  //Random number generator seed 1
   Double_t   fSeed2;  //Random number generator seed 2

public:
   TRandom2(UInt_t seed=65539);
   virtual ~TRandom2();
   virtual  void     GetSeed2(UInt_t &seed1, UInt_t &seed2);
   virtual  Double_t Rndm(Int_t i=0);
   virtual  void     SetSeed(UInt_t seed=0);
   virtual  void     SetSeed2(UInt_t seed1, UInt_t seed2);

   ClassDef(TRandom2,1)  //Random number generators with periodicity > 10**14
};

R__EXTERN TRandom *gRandom;

#endif
