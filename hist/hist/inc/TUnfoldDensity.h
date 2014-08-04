// Author: Stefan Schmitt
// DESY, 11/08/11

//  Version 17.1, add scan type RhoSquare
//
//  History:
//     Version 17.0, support for density regularisation and complex binning schemes

#ifndef ROOT_TUnfoldDensity
#define ROOT_TUnfoldDensity

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//  TUnfoldDensity, an extension of the class TUnfoldSys to correct for //
//  migration effects. TUnfoldDensity provides methods to deal with     //
//  multidimensional complex binning schemes and variable bin widths    //
//                                                                      //
//  Citation: S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

/*
  This file is part of TUnfold.

  TUnfold is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  TUnfold is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "TUnfoldSys.h"
#include "TUnfoldBinning.h"


class TUnfoldDensity : public TUnfoldSys {
 protected:
   const TUnfoldBinning * fConstOutputBins; // binning scheme for the output
   const TUnfoldBinning * fConstInputBins; // binning scheme for the input
   TUnfoldBinning *fOwnedOutputBins; // output binning scheme if owner
   TUnfoldBinning *fOwnedInputBins; // input binning scheme if owner
   TUnfoldBinning *fRegularisationConditions; // binning scheme for the regularisation conditions

 public:
   enum EDensityMode {
     kDensityModeeNone=0,
     kDensityModeBinWidth=1,
     kDensityModeUser=2,
     kDensityModeBinWidthAndUser=3
   };
 protected:

   virtual TString GetOutputBinName(Int_t iBinX) const; // name a bin

   TUnfoldDensity(void); // constructor for derived classes, do nothing

   Double_t GetDensityFactor(EDensityMode densityMode,Int_t iBin) const; // density correction factor for this bin
   void RegularizeDistributionRecursive
     (const TUnfoldBinning *binning,ERegMode regmode,
      EDensityMode densityMode,const char *distribution,
      const char *axisSteering); // regularize the given binning recursively
   void RegularizeOneDistribution
     (const TUnfoldBinning *binning,ERegMode regmode,
      EDensityMode densityMode,const char *axisSteering); // regularize the distribution of one binning node

 public:
   TUnfoldDensity(const TH2 *hist_A, EHistMap histmap,
                  ERegMode regmode = kRegModeCurvature,
                  EConstraint constraint=kEConstraintArea,
                  EDensityMode densityMode=kDensityModeBinWidthAndUser,
                  const TUnfoldBinning *outputBins=0,
                  const TUnfoldBinning *inputBins=0,
                  const char *regularisationDistribution=0,
                  const char *regularisationAxisSteering="*[UOB]"); // constructor for using the histogram classes. Default regularisation is on the curvature of the bin-width normalized density, excluding underflow and overflow bins

   virtual ~ TUnfoldDensity(void); // delete data members

   void RegularizeDistribution(ERegMode regmode,EDensityMode densityMode,
                               const char *distribution,
                               const char *axisSteering); // regularize distribution(s) of the output binning scheme


   enum EScanTauMode { // scan mode of correlation scan
      kEScanTauRhoAvg =0, // average global correlation coefficient (from TUnfold::GetRhoI())
      kEScanTauRhoMax =1, // maximum global correlation coefficient (from TUnfold::GetRhoI())
      kEScanTauRhoAvgSys =2, // average global correlation coefficient (from TUnfoldSys::GetRhoItotal())
      kEScanTauRhoMaxSys =3,  // maximum global correlation coefficient (from TUnfoldSys::GetRhoItotal())
      kEScanTauRhoSquareAvg =4, // average global correlation coefficient squared (from TUnfold::GetRhoI())
      kEScanTauRhoSquareAvgSys =5 // average global correlation coefficient squared (from TUnfoldSys::GetRhoItotal())
   };

   virtual Int_t ScanTau(Int_t nPoint,Double_t tauMin,Double_t tauMax,
                         TSpline **scanResult,Int_t mode=kEScanTauRhoAvg,
                         const char *distribution=0,const char *projectionMode=0,TGraph **lCurvePlot=0,TSpline **logTauXPlot=0,TSpline **logTauYPlot=0); // scan some variable (e.g. global correlation) and find a minimum using successive calls to DoUnfold(Double_t) at various tau
   virtual Double_t GetScanVariable(Int_t mode,const char *distribution,const char *projectionMode); // calculate variable for ScanTau()

   TH1 *GetOutput(const char *histogramName,
                  const char *histogramTitle=0,const char *distributionName=0,
                  const char *projectionMode=0,Bool_t useAxisBinning=kTRUE) const;  // get unfolding result
   TH1 *GetBias(const char *histogramName,
                const char *histogramTitle=0,const char *distributionName=0,
                const char *projectionMode=0,Bool_t useAxisBinning=kTRUE) const;      // get bias
   TH1 *GetFoldedOutput(const char *histogramName,
                        const char *histogramTitle=0,
                        const char *distributionName=0,
                        const char *projectionMode=0,Bool_t useAxisBinning=kTRUE,
                        Bool_t addBgr=kFALSE) const; // get unfolding result folded back
   TH1 *GetBackground(const char *histogramName,const char *bgrSource=0,
                      const char *histogramTitle=0,
                      const char *distributionName=0,
                      const char *projectionMode=0,Bool_t useAxisBinning=kTRUE,Int_t includeError=3,
                      Bool_t clearHist=kTRUE) const; // get background source
   TH1 *GetInput(const char *histogramName,const char *histogramTitle=0,
                 const char *distributionName=0,
                 const char *projectionMode=0,Bool_t useAxisBinning=kTRUE) const;     // get unfolding input
   TH1 *GetDeltaSysSource(const char *source,
                          const char *histogramName,
                          const char *histogramTitle=0,
                          const char *distributionName=0,
                          const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get systematic shifts from one systematic source
   TH1 *GetDeltaSysBackgroundScale(const char *bgrSource,
                                   const char *histogramName,
                                   const char *histogramTitle=0,
                                   const char *distributionName=0,
                                   const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get correlated uncertainty induced by the scale uncertainty of a background source
   TH1 *GetDeltaSysTau(const char *histogramName,
                       const char *histogramTitle=0,
                       const char *distributionName=0,
                       const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get correlated uncertainty from varying tau
   TH2 *GetEmatrixSysUncorr(const char *histogramName,
                            const char *histogramTitle=0,
                            const char *distributionName=0,
                            const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get error matrix contribution from uncorrelated errors on the matrix A
   TH2 *GetEmatrixSysBackgroundUncorr(const char *bgrSource,
                                      const char *histogramName,
                                      const char *histogramTitle=0,
                                      const char *distributionName=0,
                                      const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get error matrix from uncorrelated error of one background source
   TH2 *GetEmatrixInput(const char *histogramName,
                        const char *histogramTitle=0,
                        const char *distributionName=0,
                        const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get error contribution from input vector
   TH2 *GetEmatrixTotal(const char *histogramName,
                        const char *histogramTitle=0,
                        const char *distributionName=0,
                        const char *projectionMode=0,Bool_t useAxisBinning=kTRUE); // get total error including systematic,statistical,background,tau errors
   TH1 *GetRhoIstatbgr(const char *histogramName,const char *histogramTitle=0,
                     const char *distributionName=0,
                       const char *projectionMode=0,Bool_t useAxisBinning=kTRUE,
                     TH2 **ematInv=0);      // get global correlation coefficients, stat+bgr errors only (from TUnfold)
   TH1 *GetRhoItotal(const char *histogramName,const char *histogramTitle=0,
                     const char *distributionName=0,
                     const char *projectionMode=0,Bool_t useAxisBinning=kTRUE,
                     TH2 **ematInv=0);      // get global correlation coefficients, including systematic errors (from TUnfoldSys)
   TH2 *GetRhoIJtotal(const char *histogramName,
                      const char *histogramTitle=0,
                      const char *distributionName=0,
                      const char *projectionMode=0,Bool_t useAxisBinning=kTRUE);     // get correlation coefficients
   TH2 *GetL(const char *histogramName,
             const char *histogramTitle=0,
             Bool_t useAxisBinning=kTRUE); // get regularisation matrix
   TH1 *GetLxMinusBias(const char *histogramName,const char *histogramTitle=0); // get vector L(x-bias) of regularisation conditions

   TH2 *GetProbabilityMatrix(const char *histogramName,
                             const char *histogramTitle=0,Bool_t useAxisBinning=kTRUE) const; // get matrix of probabilities

   const TUnfoldBinning *GetInputBinning(const char *distributionName=0) const; // find binning scheme for input bins
   const TUnfoldBinning *GetOutputBinning(const char *distributionName=0) const; // find binning scheme for output bins
TUnfoldBinning *GetLBinning(void) const { return fRegularisationConditions; } // binning scheme for regularisation conditions (matrix L)
   ClassDef(TUnfoldDensity, TUnfold_CLASS_VERSION) //Unfolding with densisty regularisation
};

#endif
