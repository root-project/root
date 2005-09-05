// @(#)root/hist:$Name:  $:$Id: TSpectrum.h,v 1.12 2005/06/15 10:27:36 brun Exp $
// Author: Miroslav Morhac   27/05/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum
#define ROOT_TSpectrum

/////////////////////////////////////////////////////////////////////////////
//	THIS FILE CONTAINS HEADERS FOR ADVANCED				   //
//	SPECTRA PROCESSING FUNCTIONS.	   				   //
//									   //
//	ONE-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION			   //
//	TWO-DIMENSIONAL BACKGROUND ESTIMATION FUNCTION			   //
//	ONE-DIMENSIONAL DECONVOLUTION FUNCTION				   //
//	TWO-DIMENSIONAL DECONVOLUTION FUNCTION				   //
//	ONE-DIMENSIONAL PEAK SEARCH FUNCTION				   //
//	TWO-DIMENSIONAL PEAK SEARCH FUNCTION				   //
//									   //
//	Miroslav Morhac							   //
//	Institute of Physics						   //
//	Slovak Academy of Sciences					   //
//	Dubravska cesta 9, 842 28 BRATISLAVA				   //
//	SLOVAKIA							   //
//									   //
//	email:fyzimiro@savba.sk,	 fax:+421 7 54772479		   //
//									   //
/////////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TH1
#include "TH1.h"
#endif

const int    kBackOrder2 =0;
const int    kBackOrder4 =1;
const int    kBackOrder6 =2;
const int    kBackOrder8 =3;
const int    kBackIncreasingWindow =0;
const int    kBackDecreasingWindow =1;
const Bool_t kBackExcludeCompton = kFALSE;
const Bool_t kBackIncludeCompton = kTRUE;

const int    kFitOptimChiCounts =0;
const int    kFitOptimChiFuncValues =1;
const int    kFitOptimMaxLikelihood =2;
const int    kFitAlphaHalving =0;
const int    kFitAlphaOptimal =1;
const int    kFitPower2 =2;
const int    kFitPower4 =4;
const int    kFitPower6 =6;
const int    kFitPower8 =8;
const int    kFitPower10 =10;
const int    kFitPower12 =12;
const int    kFitTaylorOrderFirst =0;
const int    kFitTaylorOrderSecond =1;
const int    kFitNumRegulCycles =100;

const int    kTransformHaar =0;
const int    kTransformWalsh =1;
const int    kTransformCos =2;
const int    kTransformSin =3;
const int    kTransformFourier =4;
const int    kTransformHartley =5;
const int    kTransformFourierWalsh =6;
const int    kTransformFourierHaar =7;
const int    kTransformWalshHaar =8;
const int    kTransformCosWalsh =9;
const int    kTransformCosHaar =10;
const int    kTransformSinWalsh =11;
const int    kTransformSinHaar =12;
const int    kTransformForward =0;
const int    kTransformInverse =1;

const int    kMaxNumberPeaks= 1000;

//note that this class does not follow the ROOT naming conventions
class TSpectrumOneDimFit{
public:
   int    fNumberPeaks;                     //input parameter, should be>0
   int    fNumberIterations;                //input parameter, should be >0
   int    fXmin;                            //first fitted channel
   int    fXmax;                            //last fitted channel
   double fAlpha;                           //convergence coefficient, input parameter, it should be positive number and <=1
   double fChi;                             //here the function returns resulting chi square
   int    fStatisticType;                   //type of statistics, possible values kFitOptimChiCounts (chi square statistics with counts as weighting coefficients), kFitOptimChiFuncValues (chi square statistics with function values as weighting coefficients),kFitOptimMaxLikelihood
   int    fAlphaOptim;                      //optimization of convergence coefficients, possible values kFitAlphaHalving, kFitAlphaOptimal, see manual
   int    fPower;                           //possible values kFitPower2,4,6,8,10,12, see manual
   int    fFitTaylor;                       //order of Taylor expansion, possible values kFitTaylorOrderFirst, kFitTaylorOrderSecond
   double fPositionInit[kMaxNumberPeaks];   //initial values of peaks positions, input parameters
   double fPositionCalc[kMaxNumberPeaks];   //calculated values of fitted positions, output parameters
   double fPositionErr[kMaxNumberPeaks];    //position errors
   Bool_t fFixPosition[kMaxNumberPeaks];    //logical vector which allows to fix appropriate positions (not fit). However they are present in the estimated functional
   double fAmpInit[kMaxNumberPeaks];        //initial values of peaks amplitudes, input parameters
   double fAmpCalc[kMaxNumberPeaks];        //calculated values of fitted amplitudes, output parameters
   double fAmpErr[kMaxNumberPeaks];         //amplitude errors
   Bool_t fFixAmp[kMaxNumberPeaks];         //logical vector which allows to fix appropriate amplitudes (not fit). However they are present in the estimated functional
   double fArea[kMaxNumberPeaks];           //calculated areas of peaks
   double fAreaErr[kMaxNumberPeaks];        //errors of peak areas
   double fSigmaInit; //sigma parameter, meaning analogical to the above given parameters, see manual
   double fSigmaCalc;
   double fSigmaErr;
   Bool_t fFixSigma;
   double fTInit;    //t parameter, meaning analogical to the above given parameters, see manual
   double fTCalc;
   double fTErr;
   Bool_t fFixT;
   double fBInit;    //b parameter, meaning analogical to the above given parameters, see manual
   double fBCalc;
   double fBErr;
   Bool_t fFixB;
   double fSInit;    //s parameter, meaning analogical to the above given parameters, see manual
   double fSCalc;
   double fSErr;
   Bool_t fFixS;
   double fA0Init;   //backgroud is estimated as a0+a1*x+a2*x*x
   double fA0Calc;
   double fA0Err;
   Bool_t fFixA0;
   double fA1Init;
   double fA1Calc;
   double fA1Err;
   Bool_t fFixA1;
   double fA2Init;
   double fA2Calc;
   double fA2Err;
   Bool_t fFixA2;
};

class TSpectrum : public TNamed {
protected:
   Int_t         fMaxPeaks;       //Maximum number of peaks to be found
   Int_t         fNPeaks;         //number of peaks found
   Float_t      *fPosition;       //!array of current peak positions
   Float_t      *fPositionX;      //!X position of peaks
   Float_t      *fPositionY;      //!Y position of peaks
   Float_t       fResolution;     //resolution of the neighboring peaks
   TH1          *fHistogram;      //resulting histogram
static Int_t     fgAverageWindow; //Average window of searched peaks
static Int_t     fgIterations;    //Maximum number of decon iterations (default=3)

public:
   TSpectrum();
   TSpectrum(Int_t maxpositions, Float_t resolution=1);
   virtual ~TSpectrum();
   virtual const char *Background(TH1 *hist,int niter, Option_t *option="goff");
   const char         *Background1(float *spectrum,int size,int niter);
   const char         *Deconvolution1(float *source,const float *resp,int size,int niter);
   TH1                *GetHistogram() const {return fHistogram;}
   Int_t               GetNPeaks() const {return fNPeaks;}
   Float_t            *GetPositionX() const {return fPositionX;}
   Float_t            *GetPositionY() const {return fPositionY;}
   virtual Int_t       Search(TH1 *hist, Double_t sigma, Option_t *option="goff", Double_t threshold=0.05);
   static void         SetAverageWindow(Int_t w=3);   //set average window
   static void         SetDeconIterations(Int_t n=3); //set max number of decon iterations
    void               SetResolution(Float_t resolution=1);

   //new functions April 2003
   const char         *Background1General(float *spectrum,int size,int number_of_iterations,int direction,int filter_order,Bool_t compton);
   const char         *Smooth1Markov(float *source, int size, int aver_window);
   const char         *Deconvolution1HighResolution(float *source,const float *resp,int size,int number_of_iterations,int number_of_repetitions,double boost);
   const char         *Deconvolution1Unfolding(float *source,const float **resp,int sizex,int sizey,int number_of_iterations);
   Int_t               Search1HighRes(float *source,float *dest, int size, float sigma, double threshold,
   Bool_t              background_remove,int decon_iterations,Bool_t markov, int aver_window);
   double              Lls(double a);

      //auxiliary functions for 1. parameter fit functions
   double Erfc(double x);
   double Derfc(double x);
   double Deramp(double i,double i0,double sigma,double t,double s,double b);
   double Deri0(double i,double amp,double i0,double sigma,double t,double s,double b);
   double Derderi0(double i,double amp,double i0,double sigma);
   double Dersigma(int num_of_fitted_peaks,double i,const double* parameter,double sigma,double t,double s,double b);
   double Derdersigma(int num_of_fitted_peaks,double i,const double* parameter,double sigma);
   double Dert(int num_of_fitted_peaks,double i,const double* parameter,double sigma,double b);
   double Ders(int num_of_fitted_peaks,double i,const double* parameter,double sigma);
   double Derb(int num_of_fitted_peaks,double i,const double* parameter,double sigma,double t,double b);
   double Dera1(double i);
   double Dera2(double i);
   double Shape(int num_of_fitted_peaks,double i,const double *parameter,double sigma,double t,double s,double b,double a0,double a1,double a2);
   double Area(double a,double sigma,double t,double b);
   double Derpa(double sigma,double t,double b);
   double Derpsigma(double a,double t,double b);
   double Derpt(double a,double sigma,double b);
   double Derpb(double a,double sigma,double t,double b);
   double Ourpowl(double a,int pw);
   void   StiefelInversion(double **a,int rozmer);

   const char* Fit1Awmi(float *source, TSpectrumOneDimFit* p,int size);
   const char* Fit1Stiefel(float *source, TSpectrumOneDimFit* p,int size);

//////////AUXILIARY FUNCTIONS FOR 1. DIMENSIONAL TRANSFORM, FILTER AND ENHANCE FUNCTIONS ////////////////////////

   void  Haar(float *working_space,int num,int direction);
   void  Walsh(float *working_space,int num);
   void  BitReverse(float *working_space,int num);
   void  Fourier(float *working_space,int num,int hartley,int direction,int zt_clear);
   void  BitReverseHaar(float *working_space,int shift,int num,int start);
   int   GeneralExe(float *working_space,int zt_clear,int num,int degree,int type);
   int   GeneralInv(float *working_space,int num,int degree,int type);

   const char *Transform1(const float *source,float *dest,int size,int type,int direction,int degree);
   const char *Filter1Zonal(const float *source,float *dest,int size,int type,int degree,int xmin, int xmax,float filter_coeff);
   const char *Enhance1(const float *source,float *dest,int size,int type,int degree,int xmin, int xmax,float enhance_coeff);

   ClassDef(TSpectrum,2)  //Peak Finder, background estimator, Deconvolution
};

#endif

