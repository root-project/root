
//////////////////////////////////////////////////////////////////////////////
//
//  TRolke
//
//  This class computes confidence intervals for the rate of a Poisson
//  in the presence of background and efficiency with a fully frequentist
//  treatment of the uncertainties in the efficiency and background estimate
//  using the profile likelihood method.
//
//      Author: Jan Conrad (CERN) 2004
//      Updated: Johan Lundberg (CERN) 2009
//
//      Copyright CERN 2004,2009           Jan.Conrad@cern.ch,
//                                     Johan.Lundberg@cern.ch
//
//  For information about the statistical meaning of the parameters
//  and the syntax, consult TRolke.cxx
//                  ------------------
//
//  Examples are found in the file Rolke.C
//  --------------------------------------
//
//////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRolke
#define ROOT_TRolke

#include "TObject.h"
#include "TMath.h"

// Class definition. This class is not intended to be used as a base class.
class TRolke : public TObject
{

private:
   Double_t fCL;         // confidence level as a fraction [0.9 for 90% ]
   Double_t fUpperLimit; // the calculated upper limit
   Double_t fLowerLimit; // the calculated lower limit
   bool fBounding;       // false for unbounded likelihood
                         // true for bounded likelihood
   Int_t fNumWarningsDeprecated1;
   Int_t fNumWarningsDeprecated2;

   /* ----------------------------------------------------------------- */
   /* These variables are set by the Set methods for the various models */
   Int_t f_x;
   Int_t f_y;
   Int_t f_z;
   Double_t f_bm;
   Double_t f_em;
   Double_t f_e;
   Int_t f_mid;
   Double_t f_sde;
   Double_t f_sdb;
   Double_t f_tau;
   Double_t f_b;
   Int_t f_m;

   /* ----------------------------------------------------------------- */
   /* Internal helper functions and methods */
   // The Calculator
   Double_t Interval(Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Double_t e, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b, Int_t m);

   // LIKELIHOOD ROUTINE
   Double_t Likelihood(Double_t mu, Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b, Int_t m, Int_t what);

   //MODEL 1
   Double_t EvalLikeMod1(Double_t mu, Int_t x, Int_t y, Int_t z, Double_t tau, Int_t m, Int_t what);
   Double_t LikeMod1(Double_t mu, Double_t b, Double_t e, Int_t x, Int_t y, Int_t z, Double_t tau, Int_t m);
   void     ProfLikeMod1(Double_t mu, Double_t &b, Double_t &e, Int_t x, Int_t y, Int_t z, Double_t tau, Int_t m);
   Double_t LikeGradMod1(Double_t e, Double_t mu, Int_t x, Int_t y, Int_t z, Double_t tau, Int_t m);

   //MODEL 2
   Double_t EvalLikeMod2(Double_t mu, Int_t x, Int_t y, Double_t em, Double_t sde, Double_t tau, Int_t what);

   Double_t LikeMod2(Double_t mu, Double_t b, Double_t e, Int_t x, Int_t y, Double_t em, Double_t tau, Double_t v);

   //MODEL 3
   Double_t EvalLikeMod3(Double_t mu, Int_t x, Double_t bm, Double_t em, Double_t sde, Double_t sdb, Int_t what);
   Double_t LikeMod3(Double_t mu, Double_t b, Double_t e, Int_t x, Double_t bm, Double_t em, Double_t u, Double_t v);

   //MODEL 4
   Double_t EvalLikeMod4(Double_t mu, Int_t x, Int_t y, Double_t tau, Int_t what);
   Double_t LikeMod4(Double_t mu, Double_t b, Int_t x, Int_t y, Double_t tau);

   //MODEL 5
   Double_t EvalLikeMod5(Double_t mu, Int_t x, Double_t bm, Double_t sdb, Int_t what);
   Double_t LikeMod5(Double_t mu, Double_t b, Int_t x, Double_t bm, Double_t u);

   //MODEL 6
   Double_t EvalLikeMod6(Double_t mu, Int_t x, Int_t z, Double_t b, Int_t m, Int_t what);
   Double_t LikeMod6(Double_t mu, Double_t b, Double_t e, Int_t x, Int_t z, Int_t m);

   //MODEL 7
   Double_t EvalLikeMod7(Double_t mu, Int_t x, Double_t em, Double_t sde, Double_t b, Int_t what);
   Double_t LikeMod7(Double_t mu, Double_t b, Double_t e, Int_t x, Double_t em, Double_t v);

   //MISC
   static Double_t EvalPolynomial(Double_t x, const Int_t coef[], Int_t N);
   static Double_t EvalMonomial(Double_t x, const Int_t coef[], Int_t N);
   Double_t LogFactorial(Int_t n);

   Double_t ComputeInterval(Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Double_t e, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b, Int_t m);

   void SetModelParameters(Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Double_t e, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b, Int_t m);

   void SetModelParameters();

   Double_t GetBackground();

public:

   /* Constructor */
   TRolke(Double_t CL = 0.9, Option_t *option = "");

   /* Destructor */
   ~TRolke() override;

   /* Get and set the Confidence Level */
   Double_t GetCL() const         {
      return fCL;
   }
   void     SetCL(Double_t CL)  {
      fCL = CL;
   }

   /* Set the Confidence Level in terms of Sigmas. */
   void SetCLSigmas(Double_t CLsigmas) {
      fCL = TMath::Erf(CLsigmas / TMath::Sqrt(2.0)) ;
   }

   // The Set methods for the different models are described in Rolke.cxx
   // model 1
   void SetPoissonBkgBinomEff(Int_t x, Int_t y, Int_t z, Double_t tau, Int_t m);

   // model 2
   void SetPoissonBkgGaussEff(Int_t x, Int_t y, Double_t em, Double_t tau, Double_t sde);

   // model 3
   void SetGaussBkgGaussEff(Int_t x, Double_t bm, Double_t em, Double_t sde, Double_t sdb);

   // model 4
   void SetPoissonBkgKnownEff(Int_t x, Int_t y, Double_t tau, Double_t e);

   // model 5
   void SetGaussBkgKnownEff(Int_t x, Double_t bm, Double_t sdb, Double_t e);

   // model 6
   void SetKnownBkgBinomEff(Int_t x, Int_t z, Int_t m, Double_t b);

   // model 7
   void SetKnownBkgGaussEff(Int_t x, Double_t em, Double_t sde, Double_t b);

   /* Deprecated interface method (read Rolke.cxx). May be removed from future releases */
   Double_t CalculateInterval(Int_t x, Int_t y, Int_t z, Double_t bm, Double_t em, Double_t e, Int_t mid, Double_t sde, Double_t sdb, Double_t tau, Double_t b, Int_t m);

   // get the upper and lower limits based on the specified model
   bool GetLimits(Double_t& low, Double_t& high);
   Double_t GetUpperLimit();
   Double_t GetLowerLimit();

   // get the upper and lower average limits
   bool GetSensitivity(Double_t& low, Double_t& high, Double_t pPrecision = 0.00001);

   // get the upper and lower limits for the outcome corresponding to
   // a given quantile.
   bool GetLimitsQuantile(Double_t& low, Double_t& high, Int_t& out_x, Double_t integral = 0.5);

   // get the upper and lower limits for the most likely outcome.
   bool GetLimitsML(Double_t& low, Double_t& high, Int_t& out_x);

   // get the value of x corresponding to rejection of the null hypothesis.
   bool GetCriticalNumber(Int_t& ncrit,Int_t maxtry=-1);

   /* Get the bounding mode flag. True activates bounded mode. Read
      TRolke.cxx and the references therein for details. */
   bool GetBounding() const {
      return fBounding;
   }

   /* Get the bounding mode flag. True activates bounded mode. Read
      TRolke.cxx and the references therein for details. */
   void SetBounding(const bool bnd) {
      fBounding = bnd;
   }

   /* Deprecated name for SetBounding. */
   void SetSwitch(bool bnd) ;

   /* Dump internals. Option is not used */
   void Print(Option_t*) const override;

   ClassDefOverride(TRolke, 2)
};

//calculate confidence limits using the Rolke method
#endif

