// @(#)root/mathcore:$Id$
// Author: Rene Brun   04/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRandomGen
#define ROOT_TRandomGen



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandomGen                                                             //
//                                                                      //
// random number generator class template con the engine                //
//                                                        
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRandom
#include "TRandom.h"
#endif

template<class Engine>
class TRandomGen : public TRandom {

protected:

   Engine  fEngine;   // random number generator engine
public:
   
   TRandomGen(ULong_t seed=1) {
      fEngine.SetSeed(seed);
      SetName(TString::Format("Random_%s",fEngine.Name().c_str() ) );
      SetTitle(TString::Format("Random number generator: %s",fEngine.Name().c_str() ));
   }
   virtual ~TRandomGen() {}
   virtual  Double_t Rndm( ) { return fEngine(); }
   virtual  void     RndmArray(Int_t n, Float_t *array) {
      for (int i = 0; i < n; ++i) array[i] = fEngine(); 
   }
   virtual  void     RndmArray(Int_t n, Double_t *array) {
            for (int i = 0; i < n; ++i) array[i] = fEngine(); 
   }
   virtual  void     SetSeed(ULong_t seed=0) {
      fEngine.SetSeed(seed);
   }

   ClassDef(TRandomGen,1)  //Generic Random number generator template on the Engine type
};

// some useful typedef
#include "Math/StdEngine.h"
#include "Math/MixMaxEngine.h"

// not working wight now for this classes
//#define  DEFINE_TEMPL_INSTANCE
#ifdef DEFINE_TEMPL_INSTANCE

extern template class  TRandomGen<ROOT::Math::MixMaxEngine<240,0>>;
extern template class TRandomGen<ROOT::Math::MixMaxEngine<256,2>>; 
extern template class TRandomGen<ROOT::Math::MixMaxEngine<256,4>>; 
extern template class TRandomGen<ROOT::Math::MixMaxEngine<17,0>>;
extern template class TRandomGen<ROOT::Math::MixMaxEngine<17,1>>;

extern template class  TRandomGen<ROOT::Math::StdEngine<std::mt19937_64> >;
extern template class  TRandomGen<ROOT::Math::StdEngine<std::ranlux48> >;

#endif

typedef TRandomGen<ROOT::Math::MixMaxEngine<240,0>> TRandomMixMax;
typedef TRandomGen<ROOT::Math::MixMaxEngine<256,2>> TRandomMixMax256;
typedef TRandomGen<ROOT::Math::MixMaxEngine<17,0>> TRandomMixMax17;
typedef TRandomGen<ROOT::Math::StdEngine<std::mt19937_64> > TRandomMT64;
typedef TRandomGen<ROOT::Math::StdEngine<std::ranlux48> > TRandomRanlux48;


#endif
