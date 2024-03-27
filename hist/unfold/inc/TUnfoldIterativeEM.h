// Author: Stefan Schmitt
// DESY, 19/10/11

//  Version 17.9

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//  TUnfold provides functionality to correct data                      //
//   for migration effects.                                             //
//                                                                      //
//  Citation: S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]        //
//                                                                      //
//  this class implements the iterative EM unfolding method             //
//   (also called D'Agostini Method or "iterative Bayesian method")     //
//  which has been "invented" independently by numnerous authors        //
//  to unfold Poisson-distributed, mutually exclusive bins.             //
//  See e.g.                                                            //
//    Richardson, W.H., Opt. Soc. Amer. A62 (1972), 55                  //
//    Lucy, L.B., Astron. J. 79 (1974), 745.                            //
//    Vardi, Y., Shepp, L.A. and Kaufman, L.,                           //
//         J. Amer. Stat. Assoc. 80 (1985), 8.                          //
//    Multhei, H.N. and Schorr, B., Nucl. Instr. Meth. A257 (1987), 371 //
//    D'Agostini, G.,  Nucl. Instr. Meth. A362 (1995), 487              //
//                                                                      //
//  The novelty with this implementation is that the number of          //
//  iterations can be chosen based on SURE                              //
//        (Stein's unbiased Risk Estimator)                             //
//  See:                                                                //
//    Tibshirani, R.J. and Rosset, S., J. Amer. Stat. Assoc. 114, 526   //
//      [arXiv:1612.09415]                                              //
//                                                                      //
// This method is there for comparison with the Tihkonov unfolding.     //
// The interface is similar to "TUnfoldDensity"                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TUnfoldInterativeEM
#define ROOT_TUnfoldInterativeEM

#include "TUnfold.h"

class TUnfoldBinning;

class TUnfoldIterativeEM : public TObject {
 public:
   TUnfoldIterativeEM(void);
   TUnfoldIterativeEM(const TH2 *hist_A, TUnfold::EHistMap histmap,
                         const TUnfoldBinning *outputBins=nullptr,
                         const TUnfoldBinning *inputBins=nullptr);
   ~TUnfoldIterativeEM() override;
   virtual void DoUnfold(Int_t numIterations);
   virtual Int_t SetInput(const TH1 *hist_y, Double_t scaleBias=1.0);
   void SubtractBackground(const TH1 *hist_bgr,const char *name,
                           Double_t scale=1.0);
   void DoUnfold(Int_t nIter,const TH1 *hist_y, Double_t scaleBias=1.0);
   virtual Int_t ScanSURE(Int_t nIterMax,
                          TGraph **SURE=nullptr,
                          TGraph **df_deviance=nullptr);
   TH1 *GetOutput(const char *histogramName,
                  const char *histogramTitle=nullptr,const char *distributionName=nullptr,
		  const char *projectionMode=nullptr,Bool_t useAxisBinning=kTRUE) const;
   TH1 *GetFoldedOutput(const char *histogramName,
                        const char *histogramTitle=nullptr,const char *distributionName=nullptr,
                        const char *projectionMode=nullptr,Bool_t useAxisBinning=kTRUE,
                        Bool_t addBgr=kFALSE) const;
   Double_t GetDeviance(void) const;
   Double_t GetDF(void) const;
   Double_t GetSURE(void) const;
 protected:
   virtual void Reset(void);
   virtual void IterateOnce(void);
   TUnfoldBinning *f_inputBins,*f_outputBins;
   const TUnfoldBinning *f_constInputBins,*f_constOutputBins;
   TMatrixD *fA;
   TVectorD *fEpsilon;
   TVectorD *fX0;
   TVectorD *fY;
   TVectorD *fBgr;
   double fScaleBias;

   Int_t fStep;
   TVectorD *fX;
   TMatrixD *fDXDY;

   ClassDefOverride(TUnfoldIterativeEM, TUnfold_CLASS_VERSION) //iterative Unfolding with scan of SURE
};

#endif
