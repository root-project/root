// @(#)root/mathcore:$Id$
// Author: Rene Brun   04/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRandom1
#define ROOT_TRandom1



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandom1                                                             //
//                                                                      //
// Ranlux random number generator class (periodicity > 10**14)          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRandom.h"

class TRandom1 : public TRandom {

protected:
  Int_t           fNskip;
  Int_t           fLuxury;
  Int_t           fIlag;
  Int_t           fJlag;
  Int_t           fCount24;
  Float_t         fFloatSeedTable[24];
  Float_t         fCarry;
  const Int_t     fIntModulus;
  static Int_t    fgNumEngines;
  static Int_t    fgMaxIndex;
  const UInt_t   *fTheSeeds;           //! Temporary array of seed values (transient)
  const Double_t  fMantissaBit24;
  const Double_t  fMantissaBit12;

public:
   TRandom1();
   TRandom1(UInt_t seed, Int_t lux = 3 );
   TRandom1(Int_t rowIndex, Int_t colIndex, Int_t lux );
   ~TRandom1() override;
   virtual  Int_t    GetLuxury() const {return fLuxury;}
                    // Get the current seed (first element of the table)
    UInt_t   GetSeed() const override { return  UInt_t ( fFloatSeedTable[0] /  fMantissaBit24 ) ; }
                    // Gets the current seed.
   const UInt_t     *GetTheSeeds() const {return fTheSeeds;}
                     // Gets the current array of seeds.
   static   void     GetTableSeeds(UInt_t* seeds, Int_t index);
                     // Gets back seed values stored in the table, given the index.
   using TRandom::Rndm;
    Double_t Rndm( ) override;
    void     RndmArray(Int_t size, Float_t *vect) override;
    void     RndmArray(Int_t size, Double_t *vect) override;
   virtual  void     SetSeed2(UInt_t seed, Int_t lux=3);
                     // Sets the state of the algorithm according to seed.
   virtual  void     SetSeeds(const UInt_t * seeds, Int_t lux=3);
                     // Sets the state of the algorithm according to the zero terminated
                     // array of seeds. Only the first seed is used.
    void     SetSeed(ULong_t seed) override;

   ClassDefOverride(TRandom1,2)  //Ranlux Random number generators with periodicity > 10**14
};

R__EXTERN TRandom *gRandom;

#endif
