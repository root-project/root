// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrumFit
#define ROOT_TSpectrumFit

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpectrumFit                                                         //
//                                                                      //
// Class for fitting 1D spectra using AWMI (algorithm without matrix    //
// inversion) and conjugate gradient algorithms for symmetrical         //
// matrices (Stiefel-Hestens method). AWMI method allows to fit         //
// simulaneously 100s up to 1000s peaks. Stiefel method is very stable, //
// it converges faster, but is more time consuming                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TH1;

class TSpectrumFit : public TNamed {
protected:
   Int_t     fNPeaks;                    //number of peaks present in fit, input parameter, it should be > 0
   Int_t     fNumberIterations;          //number of iterations in fitting procedure, input parameter, it should be > 0
   Int_t     fXmin;                      //first fitted channel
   Int_t     fXmax;                      //last fitted channel
   Int_t     fStatisticType;             //type of statistics, possible values kFitOptimChiCounts (chi square statistics with counts as weighting coefficients), kFitOptimChiFuncValues (chi square statistics with function values as weighting coefficients),kFitOptimMaxLikelihood
   Int_t     fAlphaOptim;                //optimization of convergence algorithm, possible values kFitAlphaHalving, kFitAlphaOptimal
   Int_t     fPower;                     //possible values kFitPower2,4,6,8,10,12, for details see references. It applies only for Awmi fitting function.
   Int_t     fFitTaylor;                 //order of Taylor expansion, possible values kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi fitting function.
   Double_t  fAlpha;                     //convergence coefficient, input parameter, it should be positive number and <=1, for details see references
   Double_t  fChi;                       //here the fitting functions return resulting chi square   
   Double_t *fPositionInit;              //[fNPeaks] array of initial values of peaks positions, input parameters
   Double_t *fPositionCalc;              //[fNPeaks] array of calculated values of fitted positions, output parameters
   Double_t *fPositionErr;               //[fNPeaks] array of position errors
   Double_t *fAmpInit;                   //[fNPeaks] array of initial values of peaks amplitudes, input parameters
   Double_t *fAmpCalc;                   //[fNPeaks] array of calculated values of fitted amplitudes, output parameters
   Double_t *fAmpErr;                    //[fNPeaks] array of amplitude errors
   Double_t *fArea;                      //[fNPeaks] array of calculated areas of peaks
   Double_t *fAreaErr;                   //[fNPeaks] array of errors of peak areas
   Double_t  fSigmaInit;                 //initial value of sigma parameter
   Double_t  fSigmaCalc;                 //calculated value of sigma parameter
   Double_t  fSigmaErr;                  //error value of sigma parameter
   Double_t  fTInit;                     //initial value of t parameter (relative amplitude of tail), for details see html manual and references
   Double_t  fTCalc;                     //calculated value of t parameter
   Double_t  fTErr;                      //error value of t parameter
   Double_t  fBInit;                     //initial value of b parameter (slope), for details see html manual and references
   Double_t  fBCalc;                     //calculated value of b parameter
   Double_t  fBErr;                      //error value of b parameter
   Double_t  fSInit;                     //initial value of s parameter (relative amplitude of step), for details see html manual and references
   Double_t  fSCalc;                     //calculated value of s parameter
   Double_t  fSErr;                      //error value of s parameter
   Double_t  fA0Init;                    //initial value of background a0 parameter(backgroud is estimated as a0+a1*x+a2*x*x)
   Double_t  fA0Calc;                    //calculated value of background a0 parameter
   Double_t  fA0Err;                     //error value of background a0 parameter
   Double_t  fA1Init;                    //initial value of background a1 parameter(backgroud is estimated as a0+a1*x+a2*x*x)
   Double_t  fA1Calc;                    //calculated value of background a1 parameter
   Double_t  fA1Err;                     //error value of background a1 parameter
   Double_t  fA2Init;                    //initial value of background a2 parameter(backgroud is estimated as a0+a1*x+a2*x*x)
   Double_t  fA2Calc;                    //calculated value of background a2 parameter
   Double_t  fA2Err;                     //error value of background a2 parameter
   Bool_t   *fFixPosition;               //[fNPeaks] array of logical values which allow to fix appropriate positions (not fit). However they are present in the estimated functional   
   Bool_t   *fFixAmp;                    //[fNPeaks] array of logical values which allow to fix appropriate amplitudes (not fit). However they are present in the estimated functional      
   Bool_t    fFixSigma;                  //logical value of sigma parameter, which allows to fix the parameter (not to fit).   
   Bool_t    fFixT;                      //logical value of t parameter, which allows to fix the parameter (not to fit).      
   Bool_t    fFixB;                      //logical value of b parameter, which allows to fix the parameter (not to fit).   
   Bool_t    fFixS;                      //logical value of s parameter, which allows to fix the parameter (not to fit).      
   Bool_t    fFixA0;                     //logical value of a0 parameter, which allows to fix the parameter (not to fit).
   Bool_t    fFixA1;                     //logical value of a1 parameter, which allows to fix the parameter (not to fit).   
   Bool_t    fFixA2;                     //logical value of a2 parameter, which allows to fix the parameter (not to fit).

public:
   enum {
       kFitOptimChiCounts =0,
       kFitOptimChiFuncValues =1,
       kFitOptimMaxLikelihood =2,
       kFitAlphaHalving =0,
       kFitAlphaOptimal =1,
       kFitPower2 =2,
       kFitPower4 =4,
       kFitPower6 =6,
       kFitPower8 =8,
       kFitPower10 =10,
       kFitPower12 =12,
       kFitTaylorOrderFirst =0,
       kFitTaylorOrderSecond =1,
       kFitNumRegulCycles =100
   };
   TSpectrumFit(void); //default constructor     
   TSpectrumFit(Int_t numberPeaks); 
   virtual ~TSpectrumFit();

   //auxiliary functions for 1. parameter fit functions
protected:   
   Double_t            Area(Double_t a,Double_t sigma,Double_t t,Double_t b);
   Double_t            Dera1(Double_t i);
   Double_t            Dera2(Double_t i);
   Double_t            Deramp(Double_t i,Double_t i0,Double_t sigma,Double_t t,Double_t s,Double_t b);   
   Double_t            Derb(Int_t num_of_fitted_peaks,Double_t i,const Double_t* parameter,Double_t sigma,Double_t t,Double_t b);
   Double_t            Derderi0(Double_t i,Double_t amp,Double_t i0,Double_t sigma);
   Double_t            Derdersigma(Int_t num_of_fitted_peaks,Double_t i,const Double_t* parameter,Double_t sigma);
   Double_t            Derfc(Double_t x);
   Double_t            Deri0(Double_t i,Double_t amp,Double_t i0,Double_t sigma,Double_t t,Double_t s,Double_t b);   
   Double_t            Derpa(Double_t sigma,Double_t t,Double_t b);
   Double_t            Derpb(Double_t a,Double_t sigma,Double_t t,Double_t b);
   Double_t            Derpsigma(Double_t a,Double_t t,Double_t b);
   Double_t            Derpt(Double_t a,Double_t sigma,Double_t b);
   Double_t            Ders(Int_t num_of_fitted_peaks,Double_t i,const Double_t* parameter,Double_t sigma);
   Double_t            Dersigma(Int_t num_of_fitted_peaks,Double_t i,const Double_t* parameter,Double_t sigma,Double_t t,Double_t s,Double_t b);
   Double_t            Dert(Int_t num_of_fitted_peaks,Double_t i,const Double_t* parameter,Double_t sigma,Double_t b);
   Double_t            Erfc(Double_t x);
   Double_t            Ourpowl(Double_t a,Int_t pw);   
   Double_t            Shape(Int_t num_of_fitted_peaks,Double_t i,const Double_t *parameter,Double_t sigma,Double_t t,Double_t s,Double_t b,Double_t a0,Double_t a1,Double_t a2);
   void                StiefelInversion(Double_t **a,Int_t rozmer);

public:
   void                FitAwmi(float *source); 
   void                FitStiefel(float *source); 
   Double_t           *GetAmplitudes() const {return fAmpCalc;}   
   Double_t           *GetAmplitudesErrors() const {return fAmpErr;}
   Double_t           *GetAreas() const {return fArea;}            
   Double_t           *GetAreasErrors() const {return fAreaErr;}               
   void                GetBackgroundParameters(Double_t &a0, Double_t &a0Err, Double_t &a1, Double_t &a1Err, Double_t &a2, Double_t &a2Err);
   Double_t            GetChi() const {return fChi;}
   Double_t           *GetPositions() const {return fPositionCalc;}
   Double_t           *GetPositionsErrors() const {return fPositionErr;}   
   void                GetSigma(Double_t &sigma, Double_t &sigmaErr);
   void                GetTailParameters(Double_t &t, Double_t &tErr, Double_t &b, Double_t &bErr, Double_t &s, Double_t &sErr);
   void                SetBackgroundParameters(Double_t a0Init, Bool_t fixA0, Double_t a1Init, Bool_t fixA1, Double_t a2Init, Bool_t fixA2);
   void                SetFitParameters(Int_t xmin,Int_t xmax, Int_t numberIterations, Double_t alpha, Int_t statisticType, Int_t alphaOptim, Int_t power, Int_t fitTaylor);
   void                SetPeakParameters(Double_t sigma, Bool_t fixSigma, const Float_t *positionInit, const Bool_t *fixPosition, const Float_t *ampInit, const Bool_t *fixAmp);    
   void                SetTailParameters(Double_t tInit, Bool_t fixT, Double_t bInit, Bool_t fixB, Double_t sInit, Bool_t fixS); 

   ClassDef(TSpectrumFit,1)  //Spectrum Fitter using algorithm without matrix inversion and conjugate gradient method for symmetrical matrices (Stiefel-Hestens method)
};

#endif

