#ifndef ROOT_TRolke
#define ROOT_TRolke

#include "TObject.h"

class TRolke : public TObject {
protected:
  Double_t fCL;         // confidence level as a fraction [e.g. 90% = 0.9]
  Double_t fUpperLimit; // the calculated upper limit
  Double_t fLowerLimit; // the calculated lower limit

 // LIKELIHOOD ROUTINE

  Double_t Likelihood(Double_t mu, Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Double_t e, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b, Int_t m, Int_t what);

  //MODEL 1
  Double_t EvalLikeMod1(Double_t mu, Int_t x, Int_t y, Int_t z, Double_t e, Double_t tau, Double_t b, Int_t m, Int_t what);
  Double_t LikeMod1(Double_t mu,Double_t b, Double_t e, Int_t x, Int_t y, Int_t z, Double_t tau, Int_t m);
  void     ProfLikeMod1(Double_t mu,Double_t &b, Double_t &e,Int_t x,Int_t y, Int_t z,Double_t tau,Int_t m);
  Double_t LikeGradMod1(Double_t e, Double_t mu, Int_t x,Int_t y,Int_t z,Double_t tau,Int_t m);
  
  //MODEL 2
  Double_t EvalLikeMod2(Double_t mu, Int_t x, Int_t y, Double_t em, Double_t e,Double_t sde, Double_t tau, Double_t b, Int_t what);

  Double_t LikeMod2(Double_t mu, Double_t b, Double_t e,Int_t x,Int_t y,Double_t em,Double_t tau, Double_t v);
    
  //MODEL 3
  Double_t EvalLikeMod3(Double_t mu, Int_t x, Double_t bm, Double_t em, Double_t e, Double_t sde, Double_t sdb, Double_t b, Int_t what);  
  Double_t LikeMod3(Double_t mu,Double_t b,Double_t e,Int_t x,Double_t bm,Double_t em,Double_t u,Double_t v);
  
  //MODEL 4  
  Double_t EvalLikeMod4(Double_t mu, Int_t x, Int_t y, Double_t tau, Double_t b, Int_t what);
  Double_t LikeMod4(Double_t mu,Double_t b,Int_t x,Int_t y,Double_t tau);
  
  //MODEL 5
  Double_t EvalLikeMod5(Double_t mu, Int_t x, Double_t bm, Double_t sdb, Double_t b, Int_t what);  
  Double_t LikeMod5(Double_t mu,Double_t b,Int_t x,Double_t bm,Double_t u);
  
  //MODEL 6
  Double_t EvalLikeMod6(Double_t mu, Int_t x, Int_t z, Double_t e, Double_t b, Int_t m, Int_t what);  
  Double_t LikeMod6(Double_t mu,Double_t b,Double_t e,Int_t x,Int_t z,Int_t m);
  
  //MODEL 7
  Double_t EvalLikeMod7(Double_t mu, Int_t x, Double_t em, Double_t e, Double_t sde, Double_t b, Int_t what); 
  Double_t LikeMod7(Double_t mu,Double_t b,Double_t e,Int_t x,Double_t em,Double_t v);
 
  //MISC
  static Double_t Chi2Percentile(Double_t df,Double_t CL1);
  static Double_t InverseIncompleteGamma(Double_t df,Double_t CL1);
  static Double_t InverseNormal(Double_t CL1);
  static Double_t EvalPolynomial(Double_t x, const Int_t coef[], Int_t N);
  static Double_t EvalMonomial  (Double_t x, const Int_t coef[], Int_t N);
 
public:
  
  TRolke(Double_t CL=0.9, Option_t *option = "");
 
  virtual ~TRolke();
 
  Double_t CalculateInterval(Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Double_t e, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b,Int_t m);
  Double_t GetUpperLimit(void) const { return fUpperLimit;}
  Double_t GetLowerLimit(void) const { return fLowerLimit;}
  Double_t GetCL(void) const         { return fCL;}
  void     SetCL(Double_t CL)  { fCL = CL; }
  
  ClassDef(TRolke,1) 
};
#endif

