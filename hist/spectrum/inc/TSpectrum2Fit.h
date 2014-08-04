// @(#)root/spectrum:$Id$
// Author: Miroslav Morhac   25/09/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TSpectrum2Fit
#define ROOT_TSpectrum2Fit

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpectrum2Fit                                                        //
//                                                                      //
// Class for fitting 2D spectra using AWMI (algorithm without matrix    //
// inversion) and conjugate gradient algorithms for symmetrical         //
// matrices (Stiefel-Hestens method). AWMI method allows to fit         //
// simulaneously 100s up to 1000s peaks. Stiefel method is very stable, //
// it converges faster, but is more time consuming                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TSpectrum2Fit : public TNamed {
protected:
   Int_t     fNPeaks;                        //number of peaks present in fit, input parameter, it should be > 0
   Int_t     fNumberIterations;              //number of iterations in fitting procedure, input parameter, it should be > 0
   Int_t     fXmin;                          //first fitted channel in x direction
   Int_t     fXmax;                          //last fitted channel in x direction
   Int_t     fYmin;                          //first fitted channel in y direction
   Int_t     fYmax;                          //last fitted channel in y direction
   Int_t     fStatisticType;                 //type of statistics, possible values kFitOptimChiCounts (chi square statistics with counts as weighting coefficients), kFitOptimChiFuncValues (chi square statistics with function values as weighting coefficients),kFitOptimMaxLikelihood
   Int_t     fAlphaOptim;                    //optimization of convergence algorithm, possible values kFitAlphaHalving, kFitAlphaOptimal
   Int_t     fPower;                         //possible values kFitPower2,4,6,8,10,12, for details see references. It applies only for Awmi fitting function.
   Int_t     fFitTaylor;                     //order of Taylor expansion, possible values kFitTaylorOrderFirst, kFitTaylorOrderSecond. It applies only for Awmi fitting function.
   Double_t  fAlpha;                         //convergence coefficient, input parameter, it should be positive number and <=1, for details see references
   Double_t  fChi;                           //here the fitting functions return resulting chi square
   Double_t *fPositionInitX;                 //[fNPeaks] array of initial values of x positions of 2D peaks, input parameters
   Double_t *fPositionCalcX;                 //[fNPeaks] array of calculated values of x positions of 2D peaks, output parameters
   Double_t *fPositionErrX;                  //[fNPeaks] array of error values of x positions of 2D peaks, output parameters
   Double_t *fPositionInitY;                 //[fNPeaks] array of initial values of y positions of 2D peaks, input parameters
   Double_t *fPositionCalcY;                 //[fNPeaks] array of calculated values of y positions of 2D peaks, output parameters
   Double_t *fPositionErrY;                  //[fNPeaks] array of error values of y positions of 2D peaks, output parameters
   Double_t *fPositionInitX1;                //[fNPeaks] array of initial x positions of 1D ridges, input parameters
   Double_t *fPositionCalcX1;                //[fNPeaks] array of calculated x positions of 1D ridges, output parameters
   Double_t *fPositionErrX1;                 //[fNPeaks] array of x positions errors of 1D ridges, output parameters
   Double_t *fPositionInitY1;                //[fNPeaks] array of initial y positions of 1D ridges, input parameters
   Double_t *fPositionCalcY1;                //[fNPeaks] array of calculated y positions of 1D ridges, output parameters
   Double_t *fPositionErrY1;                 //[fNPeaks] array of y positions errors of 1D ridges, output parameters
   Double_t *fAmpInit;                       //[fNPeaks] array of initial values of amplitudes of 2D peaks, input parameters
   Double_t *fAmpCalc;                       //[fNPeaks] array of calculated values of amplitudes of 2D peaks, output parameters
   Double_t *fAmpErr;                        //[fNPeaks] array of amplitudes errors of 2D peaks, output parameters
   Double_t *fAmpInitX1;                     //[fNPeaks] array of initial values of amplitudes of 1D ridges in x direction, input parameters
   Double_t *fAmpCalcX1;                     //[fNPeaks] array of calculated values of amplitudes of 1D ridges in x direction, output parameters
   Double_t *fAmpErrX1;                      //[fNPeaks] array of amplitudes errors of 1D ridges in x direction, output parameters
   Double_t *fAmpInitY1;                     //[fNPeaks] array of initial values of amplitudes of 1D ridges in y direction, input parameters
   Double_t *fAmpCalcY1;                     //[fNPeaks] array of calculated values of amplitudes of 1D ridges in y direction, output parameters
   Double_t *fAmpErrY1;                      //[fNPeaks] array of amplitudes errors of 1D ridges in y direction, output parameters
   Double_t *fVolume;                        //[fNPeaks] array of calculated volumes of 2D peaks, output parameters
   Double_t *fVolumeErr;                     //[fNPeaks] array of volumes errors of 2D peaks, output parameters
   Double_t  fSigmaInitX;                    //initial value of sigma x parameter
   Double_t  fSigmaCalcX;                    //calculated value of sigma x parameter
   Double_t  fSigmaErrX;                     //error value of sigma x parameter
   Double_t  fSigmaInitY;                    //initial value of sigma y parameter
   Double_t  fSigmaCalcY;                    //calculated value of sigma y parameter
   Double_t  fSigmaErrY;                     //error value of sigma y parameter
   Double_t  fRoInit;                        //initial value of correlation coefficient
   Double_t  fRoCalc;                        //calculated value of correlation coefficient
   Double_t  fRoErr;                         //error value of correlation coefficient
   Double_t  fTxyInit;                       //initial value of t parameter for 2D peaks (relative amplitude of tail), for details see html manual and references
   Double_t  fTxyCalc;                       //calculated value of t parameter for 2D peaks
   Double_t  fTxyErr;                        //error value of t parameter for 2D peaks
   Double_t  fSxyInit;                       //initial value of s parameter for 2D peaks (relative amplitude of step), for details see html manual and references
   Double_t  fSxyCalc;                       //calculated value of s parameter for 2D peaks
   Double_t  fSxyErr;                        //error value of s parameter for 2D peaks
   Double_t  fTxInit;                        //initial value of t parameter for 1D ridges in x direction (relative amplitude of tail), for details see html manual and references
   Double_t  fTxCalc;                        //calculated value of t parameter for 1D ridges in x direction
   Double_t  fTxErr;                         //error value of t parameter for 1D ridges in x direction
   Double_t  fTyInit;                        //initial value of t parameter for 1D ridges in y direction (relative amplitude of tail), for details see html manual and references
   Double_t  fTyCalc;                        //calculated value of t parameter for 1D ridges in y direction
   Double_t  fTyErr;                         //error value of t parameter for 1D ridges in y direction
   Double_t  fSxInit;                        //initial value of s parameter for 1D ridges in x direction (relative amplitude of step), for details see html manual and references
   Double_t  fSxCalc;                        //calculated value of s parameter for 1D ridges in x direction
   Double_t  fSxErr;                         //error value of s parameter for 1D ridges in x direction
   Double_t  fSyInit;                        //initial value of s parameter for 1D ridges in y direction (relative amplitude of step), for details see html manual and references
   Double_t  fSyCalc;                        //calculated value of s parameter for 1D ridges in y direction
   Double_t  fSyErr;                         //error value of s parameter for 1D ridges in y direction
   Double_t  fBxInit;                        //initial value of b parameter for 1D ridges in x direction (slope), for details see html manual and references
   Double_t  fBxCalc;                        //calculated value of b parameter for 1D ridges in x direction
   Double_t  fBxErr;                         //error value of b parameter for 1D ridges in x direction
   Double_t  fByInit;                        //initial value of b parameter for 1D ridges in y direction (slope), for details see html manual and references
   Double_t  fByCalc;                        //calculated value of b parameter for 1D ridges in y direction
   Double_t  fByErr;                         //error value of b parameter for 1D ridges in y direction
   Double_t  fA0Init;                        //initial value of background a0 parameter(backgroud is estimated as a0+ax*x+ay*y)
   Double_t  fA0Calc;                        //calculated value of background a0 parameter
   Double_t  fA0Err;                         //error value of background a0 parameter
   Double_t  fAxInit;                        //initial value of background ax parameter(backgroud is estimated as a0+ax*x+ay*y)
   Double_t  fAxCalc;                        //calculated value of background ax parameter
   Double_t  fAxErr;                         //error value of background ax parameter
   Double_t  fAyInit;                        //initial value of background ay parameter(backgroud is estimated as a0+ax*x+ay*y)
   Double_t  fAyCalc;                        //calculated value of background ay parameter
   Double_t  fAyErr;                         //error value of background ay parameter
   Bool_t   *fFixPositionX;                  //[fNPeaks] array of logical values which allow to fix appropriate x positions of 2D peaks (not fit). However they are present in the estimated functional
   Bool_t   *fFixPositionY;                  //[fNPeaks] array of logical values which allow to fix appropriate y positions of 2D peaks (not fit). However they are present in the estimated functional
   Bool_t   *fFixPositionX1;                 //[fNPeaks] array of logical values which allow to fix appropriate x positions of 1D ridges (not fit). However they are present in the estimated functional
   Bool_t   *fFixPositionY1;                 //[fNPeaks] array of logical values which allow to fix appropriate y positions of 1D ridges (not fit). However they are present in the estimated functional
   Bool_t   *fFixAmp;                        //[fNPeaks] array of logical values which allow to fix appropriate amplitudes of 2D peaks (not fit). However they are present in the estimated functional
   Bool_t   *fFixAmpX1;                      //[fNPeaks] array of logical values which allow to fix appropriate amplitudes of 1D ridges in x direction (not fit). However they are present in the estimated functional
   Bool_t   *fFixAmpY1;                      //[fNPeaks] array of logical values which allow to fix appropriate amplitudes of 1D ridges in y direction (not fit). However they are present in the estimated functional
   Bool_t    fFixSigmaX;                     //logical value of sigma x parameter, which allows to fix the parameter (not to fit).
   Bool_t    fFixSigmaY;                     //logical value of sigma y parameter, which allows to fix the parameter (not to fit).
   Bool_t    fFixRo;                         //logical value of correlation coefficient, which allows to fix the parameter (not to fit).
   Bool_t    fFixTxy;                        //logical value of t parameter for 2D peaks, which allows to fix the parameter (not to fit).
   Bool_t    fFixSxy;                        //logical value of s parameter for 2D peaks, which allows to fix the parameter (not to fit).
   Bool_t    fFixTx;                         //logical value of t parameter for 1D ridges in x direction, which allows to fix the parameter (not to fit).
   Bool_t    fFixTy;                         //logical value of t parameter for 1D ridges in y direction, which allows to fix the parameter (not to fit).
   Bool_t    fFixSx;                         //logical value of s parameter for 1D ridges in x direction, which allows to fix the parameter (not to fit).
   Bool_t    fFixSy;                         //logical value of s parameter for 1D ridges in y direction, which allows to fix the parameter (not to fit).
   Bool_t    fFixBx;                         //logical value of b parameter for 1D ridges in x direction, which allows to fix the parameter (not to fit).
   Bool_t    fFixBy;                         //logical value of b parameter for 1D ridges in y direction, which allows to fix the parameter (not to fit).
   Bool_t    fFixA0;                         //logical value of a0 parameter, which allows to fix the parameter (not to fit).
   Bool_t    fFixAx;                         //logical value of ax parameter, which allows to fix the parameter (not to fit).
   Bool_t    fFixAy;                         //logical value of ay parameter, which allows to fix the parameter (not to fit).
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
   TSpectrum2Fit(void); //default constructor
   TSpectrum2Fit(Int_t numberPeaks);
   virtual ~TSpectrum2Fit();
   //auxiliary functions for 2. parameter fit functions
protected:
   Double_t            Deramp2(Double_t x,Double_t y,Double_t x0,Double_t y0,Double_t sigmax,Double_t sigmay,Double_t ro,Double_t txy,Double_t sxy,Double_t bx,Double_t by);
   Double_t            Derampx(Double_t x,Double_t x0,Double_t sigmax,Double_t tx,Double_t sx,Double_t bx);
   Double_t            Derbx(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t txy,Double_t tx,Double_t bx,Double_t by);
   Double_t            Derby(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t txy,Double_t ty,Double_t bx,Double_t by);
   Double_t            Derderi01(Double_t x,Double_t ax,Double_t x0,Double_t sigmax);
   Double_t            Derderi02(Double_t x,Double_t y,Double_t a,Double_t x0,Double_t y0,Double_t sigmax,Double_t sigmay,Double_t ro);
   Double_t            Derderj02(Double_t x,Double_t y,Double_t a,Double_t x0,Double_t y0,Double_t sigmax,Double_t sigmay,Double_t ro);
   Double_t            Derdersigmax(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t ro);
   Double_t            Derdersigmay(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t ro);
   Double_t            Derfc(Double_t x);
   Double_t            Deri01(Double_t x,Double_t ax,Double_t x0,Double_t sigmax,Double_t tx,Double_t sx,Double_t bx);
   Double_t            Deri02(Double_t x,Double_t y,Double_t a,Double_t x0,Double_t y0,Double_t sigmax,Double_t sigmay,Double_t ro,Double_t txy,Double_t sxy,Double_t bx,Double_t by);
   Double_t            Derj02(Double_t x,Double_t y,Double_t a,Double_t x0,Double_t y0,Double_t sigmax,Double_t sigmay,Double_t ro,Double_t txy,Double_t sxy,Double_t bx,Double_t by);
   Double_t            Derpa2(Double_t sx,Double_t sy,Double_t ro);
   Double_t            Derpro(Double_t a,Double_t sx,Double_t sy,Double_t ro);
   Double_t            Derpsigmax(Double_t a,Double_t sy,Double_t ro);
   Double_t            Derpsigmay(Double_t a,Double_t sx,Double_t ro);
   Double_t            Derro(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sx,Double_t sy,Double_t r);
   Double_t            Dersigmax(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t ro,Double_t txy,Double_t sxy,Double_t tx,Double_t sx,Double_t bx,Double_t by);
   Double_t            Dersigmay(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t ro,Double_t txy,Double_t sxy,Double_t ty,Double_t sy,Double_t bx,Double_t by);
   Double_t            Dersx(Int_t numOfFittedPeaks,Double_t x,const Double_t *parameter,Double_t sigmax);
   Double_t            Dersxy(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay);
   Double_t            Dersy(Int_t numOfFittedPeaks,Double_t x,const Double_t *parameter,Double_t sigmax);
   Double_t            Dertx(Int_t numOfFittedPeaks,Double_t x,const Double_t *parameter,Double_t sigmax,Double_t bx);
   Double_t            Dertxy(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t bx,Double_t by);
   Double_t            Derty(Int_t numOfFittedPeaks,Double_t x,const Double_t *parameter,Double_t sigmax,Double_t bx);
   Double_t            Erfc(Double_t x);
   Double_t            Ourpowl(Double_t a,Int_t pw);
   Double_t            Shape2(Int_t numOfFittedPeaks,Double_t x,Double_t y,const Double_t *parameter,Double_t sigmax,Double_t sigmay,Double_t ro,Double_t a0,Double_t ax,Double_t ay,Double_t txy,Double_t sxy,Double_t tx,Double_t ty,Double_t sx,Double_t sy,Double_t bx,Double_t by);
   void                StiefelInversion(Double_t **a,Int_t size);
   Double_t            Volume(Double_t a,Double_t sx,Double_t sy,Double_t ro);

public:
   void                FitAwmi(Double_t **source);
   void                FitStiefel(Double_t **source);
   void                GetAmplitudes(Double_t *amplitudes, Double_t *amplitudesX1, Double_t *amplitudesY1);
   void                GetAmplitudeErrors(Double_t *amplitudeErrors, Double_t *amplitudeErrorsX1, Double_t *amplitudeErrorsY1);
   void                GetBackgroundParameters(Double_t &a0, Double_t &a0Err, Double_t &ax, Double_t &axErr, Double_t &ay, Double_t &ayErr);
   Double_t            GetChi() const {return fChi;}
   void                GetPositions(Double_t *positionsX, Double_t *positionsY, Double_t *positionsX1, Double_t *positionsY1);
   void                GetPositionErrors(Double_t *positionErrorsX, Double_t *positionErrorsY, Double_t *positionErrorsX1, Double_t *positionErrorsY1);
   void                GetRo(Double_t &ro, Double_t &roErr);
   void                GetSigmaX(Double_t &sigmaX, Double_t &sigmaErrX);
   void                GetSigmaY(Double_t &sigmaY, Double_t &sigmaErrY);
   void                GetTailParameters(Double_t &txy, Double_t &txyErr, Double_t &tx, Double_t &txErr, Double_t &ty, Double_t &tyErr, Double_t &bx, Double_t &bxErr, Double_t &by, Double_t &byErr, Double_t &sxy, Double_t &sxyErr, Double_t &sx, Double_t &sxErr, Double_t &sy, Double_t &syErr);
   void                GetVolumes(Double_t *volumes);
   void                GetVolumeErrors(Double_t *volumeErrors);
   void                SetBackgroundParameters(Double_t a0Init, Bool_t fixA0, Double_t axInit, Bool_t fixAx, Double_t ayInit, Bool_t fixAy);
   void                SetFitParameters(Int_t xmin,Int_t xmax,Int_t ymin,Int_t ymax, Int_t numberIterations, Double_t alpha, Int_t statisticType, Int_t alphaOptim, Int_t power, Int_t fitTaylor);
   void                SetPeakParameters(Double_t sigmaX, Bool_t fixSigmaX, Double_t sigmaY, Bool_t fixSigmaY, Double_t ro, Bool_t fixRo, const Double_t *positionInitX, const Bool_t *fixPositionX, const Double_t *positionInitY, const Bool_t *fixPositionY, const Double_t *positionInitX1, const Bool_t *fixPositionX1, const Double_t *positionInitY1, const Bool_t *fixPositionY1, const Double_t *ampInit, const Bool_t *fixAmp, const Double_t *ampInitX1, const Bool_t *fixAmpX1, const Double_t *ampInitY1, const Bool_t *fixAmpY1);
   void                SetTailParameters(Double_t tInitXY, Bool_t fixTxy, Double_t tInitX, Bool_t fixTx, Double_t tInitY, Bool_t fixTy, Double_t bInitX, Bool_t fixBx, Double_t bInitY, Bool_t fixBy, Double_t sInitXY, Bool_t fixSxy, Double_t sInitX, Bool_t fixSx, Double_t sInitY, Bool_t fixSy);

   ClassDef(TSpectrum2Fit,1)  //Spectrum2 Fitter using algorithm without matrix inversion and conjugate gradient method for symmetrical matrices (Stiefel-Hestens method)
};

#endif

