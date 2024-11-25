// @(#)root/mathcore:$Id$
// Author: Rene Brun   15/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRandom
#define ROOT_TRandom

#include "Math/TRandomEngine.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRandom                                                              //
//                                                                      //
// Simple prototype random number generator class (periodicity = 10**9) //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

class TRandom : public TNamed, public ROOT::Math::TRandomEngine {
private:
   template <typename DType, class... Args_t>
   void loopWrapper(DType (TRandom::*func)(Args_t...), std::size_t n, DType *out, Args_t... args)
   {
      for (std::size_t i = 0; i < n; ++i) {
         out[i] = (this->*func)(args...);
      }
   }

protected:
   UInt_t   fSeed;  //Random number generator seed

public:
   TRandom(UInt_t seed=65539);
   ~TRandom() override;
   virtual  Int_t    Binomial(Int_t ntot, Double_t prob);
   virtual void BinomialN(Int_t n, Int_t *x, Int_t ntot, Double_t prob)
   {
      loopWrapper(&TRandom::Binomial, n, x, ntot, prob);
   }
   virtual  Double_t BreitWigner(Double_t mean=0, Double_t gamma=1);
   virtual void BreitWignerN(Int_t n, Double_t *x, Double_t mean = 0, Double_t gamma = 1)
   {
      loopWrapper(&TRandom::BreitWigner, n, x, mean, gamma);
   }
   virtual  void     Circle(Double_t &x, Double_t &y, Double_t r);
   virtual  Double_t Exp(Double_t tau);
   virtual void ExpN(Int_t n, Double_t *x, Double_t tau) { loopWrapper(&TRandom::Exp, n, x, tau); }
   virtual  Double_t Gaus(Double_t mean=0, Double_t sigma=1);
   virtual void GausN(Int_t n, Double_t *x, Double_t mean = 0, Double_t sigma = 1)
   {
      loopWrapper(&TRandom::Gaus, n, x, mean, sigma);
   }
   virtual  UInt_t   GetSeed() const;
   virtual  UInt_t   Integer(UInt_t imax);
   virtual void IntegerN(Int_t n, UInt_t *x, UInt_t imax) { loopWrapper(&TRandom::Integer, n, x, imax); }
   virtual  Double_t Landau(Double_t mean=0, Double_t sigma=1);
   virtual void LandauN(Int_t n, Double_t *x, Double_t mean = 0, Double_t sigma = 1)
   {
      loopWrapper(&TRandom::Landau, n, x, mean, sigma);
   }
   virtual ULong64_t Poisson(Double_t mean);
   virtual void PoissonN(Int_t n, ULong64_t *x, Double_t mean) { loopWrapper(&TRandom::Poisson, n, x, mean); }
   virtual  Double_t PoissonD(Double_t mean);
   virtual void PoissonDN(Int_t n, Double_t *x, Double_t mean) { loopWrapper(&TRandom::PoissonD, n, x, mean); }
   virtual  void     Rannor(Float_t &a, Float_t &b);
   virtual  void     Rannor(Double_t &a, Double_t &b);
   virtual  void     ReadRandom(const char *filename);
   virtual  void     SetSeed(ULong_t seed=0);
    Double_t Rndm() override;
   // keep for backward compatibility
   virtual  Double_t Rndm(Int_t ) { return Rndm(); }
   virtual  void     RndmArray(Int_t n, Float_t *array);
   virtual  void     RndmArray(Int_t n, Double_t *array);
   virtual  void     Sphere(Double_t &x, Double_t &y, Double_t &z, Double_t r);
   virtual  Double_t Uniform(Double_t x1=1);
   virtual void UniformN(Int_t n, Double_t *x, Double_t x1 = 1) { loopWrapper(&TRandom::Uniform, n, x, x1); }
   virtual  Double_t Uniform(Double_t x1, Double_t x2);
   virtual void UniformN(Int_t n, Double_t *x, Double_t x1, Double_t x2)
   {
      loopWrapper(&TRandom::Uniform, n, x, x1, x2);
   }
   virtual  void     WriteRandom(const char *filename) const;

   ClassDefOverride(TRandom,3)  //Simple Random number generator (periodicity = 10**9)
};

R__EXTERN TRandom *gRandom;

#endif
