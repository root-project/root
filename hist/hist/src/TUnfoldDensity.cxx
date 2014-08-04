// Author: Stefan Schmitt, Amnon Harel
// DESY and CERN, 11/08/11

//  Version 17.1, add scan type RhoSquare, small bug fixes with useAxisBinning
//
//  History:
//    Version 17.0, support for density regularisation, complex binning schemes, tau scan

//////////////////////////////////////////////////////////////////////////
//
//  TUnfoldDensity : public TUnfoldSys : public TUnfold
//
//  TUnfold is used to decompose a measurement y into several sources x
//  given the measurement uncertainties and a matrix of migrations A
//
//  More details are described with the documentation of TUnfold.
//
//  For most applications, it is best to use TUnfoldDensity
//  instead of using TUnfoldSys or TUnfold
//
//  If you use this software, please consider the following citation
//       S.Schmitt, JINST 7 (2012) T10003 [arXiv:1205.6201]
//
//  More documentation and updates are available on
//      http://www.desy.de/~sschmitt
//
//
//  As compared to TUnfold, TUndolfDensity adds the following functionality
//    * background subtraction (see documentation of TUnfoldSys)
//    * error propagation (see documentation of TUnfoldSys)
//    * regularisation schemes respecting the bin widths
//    * support for complex, multidimensional input distributions
//
//  Complex binning schemes are imposed on the measurements y and
//  on the result vector x with the help of the class TUnfoldBinning
//  The components of x or y are part of multi-dimensional distributions.
//  The bin widths along the relevant directions in these distributions
//  are used to calculate bin densities (number of events divided by bin width)
//  or to calculate derivatives taking into account the proper distance of
//  adjacent bin centers
//
//  Complex binning schemes
//  =======================
//  in literature on unfolding, the "standard" test case is a
//  one-dimensional distribution without underflow or overflow bins.
//  The migration matrix is almost diagonal.
//
//  This "standard" case is rarely realized for real problems.
//
//  Often one has to deal with multi-dimensional input distributions.
//  In addition, there are underflow and overflow bins
//  or other background bins, possibly determined with the help of auxillary
//  measurements
//
//  In TUnfoldDensity, such complex binning schemes are handled with the help
//  of the class TUnfoldBinning. For each vector there is a tree
//  structure. The tree nodes hold multi-dimensiopnal distributions
//
//  For example, the "measurement" tree could have two leaves, one for
//  the primary distribution and one for auxillary measurements
//
//  Similarly, the "truth" tree could have two leaves, one for the
//  signal and one for the background.
//
//  each of the leaves may then have a multi-dimensional distribution.
//
//  The class TUnfoldBinning takes care to map all bins of the
//  "measurement" to the one-dimensional vector y.
//  Similarly, the "truth" bins are mapped to the vector x.
//
//  Choice of the regularisation
//  ============================
//  In TUnfoldDensity, two methods are implemented to determine tau**2
//    (1)  ScanLcurve()  locate the tau where the L-curve plot has a "kink"
//      this function is implemented in the TUnfold class
//    (2)  ScanTau() finds the solution such that some variable
//           (e.g. global correlation coefficient) is minimized
//      this function is implemented in the TUnfoldDensity class,
//      such that the variable could be made depend on the binning scheme
//
//  Each of the algorithms has its own advantages and disadvantages
//
//  The algorithm (1) does not work if the input data are too similar to the
//  MC prediction, that is unfolding with tau=0 gives a least-square sum
//  of zero. Typical no-go cases of the L-curve scan are:
//    (a) the number of measurements is too small (e.g. ny=nx)
//    (b) the input data have no statistical fluctuations
//         [identical MC events are used to fill the matrix of migrations
//          and the vector y]
//
//  The algorithm (2) only works if the variable does have a real minimum
//  as a function of tau.
//  If global correlations are minimized, the situation is as follows:
//  The matrix of migration typically introduces negative correlations.
//   The area constraint introduces some positive correlation.
//   Regularisation on the "size" introduces no correlation.
//   Regularisation on 1st or 2nd derivatives adds positive correlations.
//   For this reason, "size" regularisation does not work well with
//   the tau-scan: the higher tau, the smaller rho, but there is no minimum.
//   In contrast, the tau-scan is expected to work well with 1st or 2nd
//   derivative regularisation, because at some point the negative
//   correlations from migrations are approximately cancelled by the
//   positive correlations from the regularisation conditions.
//
//  whichever algorithm is used, the output has to be checked:
//  (1) The L-curve should have approximate L-shape
//       and the final choice of tau should not be at the very edge of the
//       scanned region
//  (2) The scan result should have a well-defined minimum and the
//       final choice of tau should sit right in the minimum
//
////////////////////////////////////////////////////////////////////////////

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

#include "TUnfoldDensity.h"
#include <TMath.h>
#include <TVectorD.h>
#include <TObjString.h>
#include <iostream>
#include <map>

// #define DEBUG

ClassImp(TUnfoldDensity)

TUnfoldDensity::TUnfoldDensity(void)
{
   // empty constructor, for derived classes
   fConstOutputBins=0;
   fConstInputBins=0;
   fOwnedOutputBins=0;
   fOwnedInputBins=0;
   fRegularisationConditions=0;
}

TUnfoldDensity::~TUnfoldDensity(void)
{
   // clean up
   if(fOwnedOutputBins) delete fOwnedOutputBins;
   if(fOwnedInputBins) delete fOwnedInputBins;
   if(fRegularisationConditions) delete fRegularisationConditions;
}

TUnfoldDensity::TUnfoldDensity
(const TH2 *hist_A, EHistMap histmap,ERegMode regmode,EConstraint constraint,
 EDensityMode densityMode,const TUnfoldBinning *outputBins,
 const TUnfoldBinning *inputBins,const char *regularisationDistribution,
 const char *regularisationAxisSteering) :
   TUnfoldSys(hist_A,histmap,kRegModeNone,constraint)
{
   // set up unfolding matrix and regularisation scheme
   //    hist_A:  matrix that describes the migrations
   //    histmap: mapping of the histogram axes to the unfolding output
   //    regmode: global regularisation mode
   //    constraint: type of constraint to use
   //    regularisationSteering: detailed steering for the regularisation
   //                  see method RegularizeDistribution()
   //    outputBins: binning scheme of the output bins
   //    inputBins: binning scheme of the input bins

   fRegularisationConditions=0;
   // set up binning schemes
   fConstOutputBins = outputBins;
   fOwnedOutputBins = 0;
   TAxis const *genAxis,*detAxis;
   if(histmap==kHistMapOutputHoriz) {
      genAxis=hist_A->GetXaxis();
      detAxis=hist_A->GetYaxis();
   } else {
      genAxis=hist_A->GetYaxis();
      detAxis=hist_A->GetXaxis();
   }
   if(!fConstOutputBins) {
      // underflow and overflow are included in the binning scheme
      // They may be used on generator level
      fOwnedOutputBins =
         new TUnfoldBinning(*genAxis,1,1);
      fConstOutputBins = fOwnedOutputBins;
   }
   // check whether binning scheme is valid
   if(fConstOutputBins->GetParentNode()) {
      Error("TUnfoldDensity",
            "Invalid output binning scheme (node is not the root node)");
   }
   fConstInputBins = inputBins;
   fOwnedInputBins = 0;
   if(!fConstInputBins) {
      // underflow and overflow are not included in the binning scheme
      // They are still used to count events which have not been reconstructed
      fOwnedInputBins =
         new TUnfoldBinning(*detAxis,0,0);
      fConstInputBins = fOwnedInputBins;
   }
   if(fConstInputBins->GetParentNode()) {
      Error("TUnfoldDensity",
            "Invalid input binning scheme (node is not the root node)");
   }
   // check whether binning scheme matches with the histogram
   // in terms of total number of bins
   Int_t nOut=genAxis->GetNbins();
   Int_t nOutMapped=TMath::Abs(fConstOutputBins->GetTH1xNumberOfBins());
   if(nOutMapped!= nOut) {
      Error("TUnfoldDensity",
            "Output binning incompatible number of bins %d!=%d",
            nOutMapped, nOut);
   }
   // check whether binning scheme matches with the histogram
   Int_t nInput=detAxis->GetNbins();
   Int_t nInputMapped=TMath::Abs(fConstInputBins->GetTH1xNumberOfBins());
   if(nInputMapped!= nInput) {
      Error("TUnfoldDensity",
            "Input binning incompatible number of bins %d!=%d ",
            nInputMapped, nInput);
   }

   // report detailed list of excluded bins
   for (Int_t ix = 0; ix <= nOut+1; ix++) {
      if(fHistToX[ix]<0) {
         Info("TUnfold","*NOT* unfolding bin %s",GetOutputBinName(ix).Data());
      }
   }

   // set up the regularisation here
   if(regmode !=kRegModeNone) {
      RegularizeDistribution
      (regmode,densityMode,regularisationDistribution,
       regularisationAxisSteering);
   }
}

TString TUnfoldDensity::GetOutputBinName(Int_t iBinX) const {
   if(!fConstOutputBins) return TUnfold::GetOutputBinName(iBinX);
   else return fConstOutputBins->GetBinName(iBinX);
}

Double_t TUnfoldDensity::GetDensityFactor
(EDensityMode densityMode,Int_t iBin) const
{
   // density correction factor for a given bin
   //    distributionName : name of the distribution within the output binning
   //    densityFlags : type of factor to calculate
   //    iBin : bin number
   Double_t factor=1.0;
   if((densityMode == kDensityModeBinWidth)||
      (densityMode == kDensityModeBinWidthAndUser)) {
      Double_t binSize=fConstOutputBins->GetBinSize(iBin);
      if(binSize>0.0) factor /= binSize;
      else factor=0.0;
   }
   if((densityMode == kDensityModeUser)||
      (densityMode == kDensityModeBinWidthAndUser)) {
      factor *= fConstOutputBins->GetBinFactor(iBin);
   }
   return factor;
}

void TUnfoldDensity::RegularizeDistribution
(ERegMode regmode,EDensityMode densityMode,const char *distribution,
 const char *axisSteering)
{
   // regularize distribution(s) using the given settings
   //     regmode: basic regularisation mode (see class TUnfold)
   //     densityMode: how to apply bin density corrections
   //              (normalisation to bin width or user factor)
   //     distribution: name of the distribiution where this regularisation
   //             is applied to (if zero, apply to all)
   //     axisSteering: regularisation steering specific to the axes
   //          The steering is defined as follows
   //             "steering1;steering2;...steeringN"
   //          each "steeringX" is defined as
   //             axisName:[options]
   //          axisName: the name of an axis where "options" applies
   //                    the special name * matches all axes
   //          options: one of several character as follows
   //             u : exclude underflow bin from derivatives along this axis
   //             o : exclude overflow bin from derivatives along this axis
   //             U : exclude underflow bin
   //             O : exclude overflow bin
   //             b : use bin width for derivative calculation
   //             B : same as 'b' but in addition normalize to average bin width
   //
   //          example:  "*[UOB]" uses bin widths for derivatives and
   //                             underflow/overflow bins are not regularized

   RegularizeDistributionRecursive(GetOutputBinning(),regmode,densityMode,
                                   distribution,axisSteering);
}

void TUnfoldDensity::RegularizeDistributionRecursive
(const TUnfoldBinning *binning,ERegMode regmode,
 EDensityMode densityMode,const char *distribution,const char *axisSteering) {
   // recursively regularize distribution(s) using the given settings
   //     binning: distributions for this node an its children are considered
   //     regmode: basic regularisation mode (see class TUnfold)
   //     densityMode: how to apply bin density corrections
   //              (normalisation to bin withd or user factor)
   //     distribution: name of the distribiution where this regularisation
   //             is applied to (if zero, apply to all)
   //     axisSteering: regularisation steering specific to the axes
   //              (see method RegularizeDistribution())
   if((!distribution)|| !TString(distribution).CompareTo(binning->GetName())) {
      RegularizeOneDistribution(binning,regmode,densityMode,axisSteering);
   }
   for(const TUnfoldBinning *child=binning->GetChildNode();child;
       child=child->GetNextNode()) {
      RegularizeDistributionRecursive(child,regmode,densityMode,distribution,
                                      axisSteering);
   }
}

void TUnfoldDensity::RegularizeOneDistribution
(const TUnfoldBinning *binning,ERegMode regmode,
 EDensityMode densityMode,const char *axisSteering)
{
   // regularize the distribution in this node
   //     binning: the distributions to regularize
   //     regmode: basic regularisation mode (see class TUnfold)
   //     densityMode: how to apply bin density corrections
   //              (normalisation to bin withd or user factor)
   //     axisSteering: regularisation steering specific to the axes
   //              (see method RegularizeDistribution())
   if(!fRegularisationConditions)
      fRegularisationConditions=new TUnfoldBinning("regularisation");

   TUnfoldBinning *thisRegularisationBinning=
      fRegularisationConditions->AddBinning(binning->GetName());

   // decode steering
   Int_t isOptionGiven[6] = {0};
   binning->DecodeAxisSteering(axisSteering,"uUoObB",isOptionGiven);
   // U implies u
   isOptionGiven[0] |= isOptionGiven[1];
   // O implies o
   isOptionGiven[2] |= isOptionGiven[3];
   // B implies b
   isOptionGiven[4] |= isOptionGiven[5];
#ifdef DEBUG
   cout<<" "<<isOptionGiven[0]
       <<" "<<isOptionGiven[1]
       <<" "<<isOptionGiven[2]
       <<" "<<isOptionGiven[3]
       <<" "<<isOptionGiven[4]
       <<" "<<isOptionGiven[5]
       <<"\n";
#endif
   Info("RegularizeOneDistribution","regularizing %s regMode=%d"
        " densityMode=%d axisSteering=%s",
        binning->GetName(),(Int_t) regmode,(Int_t)densityMode,
        axisSteering ? axisSteering : "");
   Int_t startBin=binning->GetStartBin();
   Int_t endBin=startBin+ binning->GetDistributionNumberOfBins();
   std::vector<Double_t> factor(endBin-startBin);
   Int_t nbin=0;
   for(Int_t bin=startBin;bin<endBin;bin++) {
      factor[bin-startBin]=GetDensityFactor(densityMode,bin);
      if(factor[bin-startBin] !=0.0) nbin++;
   }
#ifdef DEBUG
   cout<<"initial number of bins "<<nbin<<"\n";
#endif
   Int_t dimension=binning->GetDistributionDimension();

   // decide whether to skip underflow/overflow bins
   nbin=0;
   for(Int_t bin=startBin;bin<endBin;bin++) {
      Int_t uStatus,oStatus;
      binning->GetBinUnderflowOverflowStatus(bin,&uStatus,&oStatus);
      if(uStatus & isOptionGiven[1]) factor[bin-startBin]=0.;
      if(oStatus & isOptionGiven[3]) factor[bin-startBin]=0.;
      if(factor[bin-startBin] !=0.0) nbin++;
   }
#ifdef DEBUG
   cout<<"after underflow/overflow bin removal "<<nbin<<"\n";
#endif
   if(regmode==kRegModeSize) {
      Int_t nRegBins=0;
      // regularize all bins of the distribution, possibly excluding
      // underflow/overflow bins
      for(Int_t bin=startBin;bin<endBin;bin++) {
         if(factor[bin-startBin]==0.0) continue;
         if(AddRegularisationCondition(bin,factor[bin-startBin])) {
            nRegBins++;
         }
      }
      if(nRegBins) {
         thisRegularisationBinning->AddBinning("size",nRegBins);
      }
   } else if((regmode==kRegModeDerivative)||(regmode==kRegModeCurvature)) {
      for(Int_t direction=0;direction<dimension;direction++) {
         // for each direction
         Int_t nRegBins=0;
         Int_t directionMask=(1<<direction);
         Double_t binDistanceNormalisation=
            (isOptionGiven[5] & directionMask)  ?
            binning->GetDistributionAverageBinSize
            (direction,isOptionGiven[0] & directionMask,
             isOptionGiven[2] & directionMask) : 1.0;
         for(Int_t bin=startBin;bin<endBin;bin++) {
            // check whether bin is excluded
            if(factor[bin-startBin]==0.0) continue;
            // for each bin, find the neighbour bins
            Int_t iPrev,iNext;
            Double_t distPrev,distNext;
            binning->GetBinNeighbours
               (bin,direction,&iPrev,&distPrev,&iNext,&distNext);
            if((regmode==kRegModeDerivative)&&(iNext>=0)) {
               Double_t f0 = -factor[bin-startBin];
               Double_t f1 = factor[iNext-startBin];
               if(isOptionGiven[4] & directionMask) {
                  if(distNext>0.0) {
                     f0 *= binDistanceNormalisation/distNext;
                     f1 *= binDistanceNormalisation/distNext;
                  } else {
                     f0=0.;
                     f1=0.;
                  }
               }
               if((f0==0.0)||(f1==0.0)) continue;
               if(AddRegularisationCondition(bin,f0,iNext,f1)) {
                  nRegBins++;
#ifdef DEBUG
                  std::cout<<"Added Reg: bin "<<bin<<" "<<f0
                           <<" next: "<<iNext<<" "<<f1<<"\n";
#endif
               }
            } else if((regmode==kRegModeCurvature)&&(iPrev>=0)&&(iNext>=0)) {
               Double_t f0 = factor[iPrev-startBin];
               Double_t f1 = -factor[bin-startBin];
               Double_t f2 = factor[iNext-startBin];
               if(isOptionGiven[4] & directionMask) {
                  if((distPrev<0.)&&(distNext>0.)) {
                     distPrev= -distPrev;
                     Double_t f=TMath::Power(binDistanceNormalisation,2.)/
                        (distPrev+distNext);
                     f0 *= f/distPrev;
                     f1 *= f*(1./distPrev+1./distNext);
                     f2 *= f/distNext;
                  } else {
                     f0=0.;
                     f1=0.;
                     f2=0.;
                  }
               }
               if((f0==0.0)||(f1==0.0)||(f2==0.0)) continue;
               if(AddRegularisationCondition(iPrev,f0,bin,f1,iNext,f2)) {
                  nRegBins++;
#ifdef DEBUG
                  std::cout<<"Added Reg: prev "<<iPrev<<" "<<f0
                           <<" bin: "<<bin<<" "<<f1
                           <<" next: "<<iNext<<" "<<f2<<"\n";
#endif
               }
            }
         }
         if(nRegBins) {
            TString name;
            if(regmode==kRegModeDerivative) {
               name="derivative_";
            } else if(regmode==kRegModeCurvature) {
               name="curvature_";
            }
            name +=  binning->GetDistributionAxisLabel(direction);
            thisRegularisationBinning->AddBinning(name,nRegBins);
         }
      }
   }
#ifdef DEBUG
   //fLsquared->Print();
#endif
}

TH1 *TUnfoldDensity::GetOutput
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning) const
{
   // retreive unfolding result as histogram
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
     TUnfoldSys::GetOutput(r,binMap);
   }
   if(binMap) {
     delete [] binMap;
   }
   return r;
}

TH1 *TUnfoldDensity::GetBias
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning) const
{
   // retreive unfolding bias as histogram
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetBias(r,binMap);
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetFoldedOutput
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning,Bool_t addBgr) const
{
   // retreive unfolding result folded back by the matrix
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   addBgr: true if the background shall be included
   TUnfoldBinning const *binning=fConstInputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetFoldedOutput(r,binMap);
      if(addBgr) {
         TUnfoldSys::GetBackground(r,0,binMap,0,kFALSE);
      }
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetBackground
(const char *histogramName,const char *bgrSource,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,Bool_t useAxisBinning,
 Int_t includeError,Bool_t clearHist) const
{
   // retreive a background source
   //   histogramName:  name of the histogram
   //   bgrSource: name of the background source
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   include error: +1 if uncorrelated bgr errors should be included
   //                  +2 if correlated bgr errors should be included
   //   clearHist: whether the histogram should be cleared
   //              if false, the background sources are added to the histogram
   TUnfoldBinning const *binning=fConstInputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetBackground(r,bgrSource,binMap,includeError,clearHist);
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetInput
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning) const
{
   // retreive input distribution
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   TUnfoldBinning const *binning=fConstInputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetInput(r,binMap);
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetRhoItotal
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning,TH2 **ematInv) {
   // retreive global correlation coefficients, total error
   // and inverse of error matrix
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   //   ematInv: retreive inverse of error matrix
   //              if ematInv==0 the inverse is not returned
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TH2 *invEmat=0;
      if(ematInv) {
         if(r->GetDimension()==1) {
            TString ematName(histogramName);
            ematName += "_inverseEMAT";
            Int_t *binMap2D=0;
            invEmat=binning->CreateErrorMatrixHistogram
               (ematName,useAxisBinning,&binMap2D,histogramTitle,
                axisSteering);
            if(binMap2D) delete [] binMap2D;
         } else {
            Error("GetRhoItotal",
                  "can not return inverse of error matrix for this binning");
         }
      }
      TUnfoldSys::GetRhoItotal(r,binMap,invEmat);
      if(invEmat) {
         *ematInv=invEmat;
      }
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetRhoIstatbgr
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning,TH2 **ematInv) {
   // retreive global correlation coefficients, input error
   // and inverse of corresponding error matrix
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   //   ematInv: retreive inverse of error matrix
   //              if ematInv==0 the inverse is not returned
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TH2 *invEmat=0;
      if(ematInv) {
         if(r->GetDimension()==1) {
            TString ematName(histogramName);
            ematName += "_inverseEMAT";
            Int_t *binMap2D=0;
            invEmat=binning->CreateErrorMatrixHistogram
               (ematName,useAxisBinning,&binMap2D,histogramTitle,
                axisSteering);
            if(binMap2D) delete [] binMap2D;
         } else {
            Error("GetRhoItotal",
                  "can not return inverse of error matrix for this binning");
         }
      }
      TUnfoldSys::GetRhoI(r,binMap,invEmat);
      if(invEmat) {
         *ematInv=invEmat;
      }
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetDeltaSysSource
(const char *source,const char *histogramName,
 const char *histogramTitle,const char *distributionName,
 const char *axisSteering,Bool_t useAxisBinning) {
   // retreive histogram of systematic 1-sigma shifts
   //   source: name of systematic error
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      if(!TUnfoldSys::GetDeltaSysSource(r,source,binMap)) {
         delete r;
         r=0;
      }
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetDeltaSysBackgroundScale
(const char *bgrSource,const char *histogramName,
 const char *histogramTitle,const char *distributionName,
 const char *axisSteering,Bool_t useAxisBinning) {
   // retreive histogram of systematic 1-sigma shifts due to a background
   // normalisation uncertainty
   //   source: name of background source
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      if(!TUnfoldSys::GetDeltaSysBackgroundScale(r,bgrSource,binMap)) {
         delete r;
         r=0;
      }
   }
   if(binMap) delete [] binMap;
   return r;
}

TH1 *TUnfoldDensity::GetDeltaSysTau
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,Bool_t useAxisBinning)
{
   // retreive histogram of systematic 1-sigma shifts due to tau uncertainty
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH1 *r=binning->CreateHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      if(!TUnfoldSys::GetDeltaSysTau(r,binMap)) {
         delete r;
         r=0;
      }
   }
   if(binMap) delete [] binMap;
   return r;
}

TH2 *TUnfoldDensity::GetRhoIJtotal
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning)
{
   // retreive histogram of total corelation coefficients, including systematic
   // uncertainties
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TH2 *r=GetEmatrixTotal
      (histogramName,histogramTitle,distributionName,
       axisSteering,useAxisBinning);
   if(r) {
      for(Int_t i=0;i<=r->GetNbinsX()+1;i++) {
         Double_t e_i=r->GetBinContent(i,i);
         if(e_i>0.0) e_i=TMath::Sqrt(e_i);
         else e_i=0.0;
         for(Int_t j=0;j<=r->GetNbinsY()+1;j++) {
            if(i==j) continue;
            Double_t e_j=r->GetBinContent(j,j);
            if(e_j>0.0) e_j=TMath::Sqrt(e_j);
            else e_j=0.0;
            Double_t e_ij=r->GetBinContent(i,j);
            if((e_i>0.0)&&(e_j>0.0)) {
               r->SetBinContent(i,j,e_ij/e_i/e_j);
            } else {
               r->SetBinContent(i,j,0.0);
            }
         }
      }
      for(Int_t i=0;i<=r->GetNbinsX()+1;i++) {
         if(r->GetBinContent(i,i)>0.0) {
            r->SetBinContent(i,i,1.0);
         } else {
            r->SetBinContent(i,i,0.0);
         }
      }
   }
   return r;
}

TH2 *TUnfoldDensity::GetEmatrixSysUncorr
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning)
{
   // get error matrix contribution from uncorrelated errors on the matrix A
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH2 *r=binning->CreateErrorMatrixHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetEmatrixSysUncorr(r,binMap);
   }
   if(binMap) delete [] binMap;
   return r;
}


TH2 *TUnfoldDensity::GetEmatrixSysBackgroundUncorr
(const char *bgrSource,const char *histogramName,
 const char *histogramTitle,const char *distributionName,
 const char *axisSteering,Bool_t useAxisBinning)
{
   // get error matrix from uncorrelated error of one background source
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH2 *r=binning->CreateErrorMatrixHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetEmatrixSysBackgroundUncorr(r,bgrSource,binMap,kFALSE);
   }
   if(binMap) delete [] binMap;
   return r;
}

TH2 *TUnfoldDensity::GetEmatrixInput
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning)
{
   // get error contribution from input vector
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH2 *r=binning->CreateErrorMatrixHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetEmatrixInput(r,binMap);
   }
   if(binMap) delete [] binMap;
   return r;
}

TH2 *TUnfoldDensity::GetProbabilityMatrix
(const char *histogramName,const char *histogramTitle,
 Bool_t useAxisBinning) const
{
   // get matrix of probabilities
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   useAxisBinning: if true, try to get the histogram using
   //                   the original matrix binning
   TH2 *r=TUnfoldBinning::CreateHistogramOfMigrations
      (fConstOutputBins,fConstInputBins,histogramName,
       useAxisBinning,useAxisBinning,histogramTitle);
   TUnfold::GetProbabilityMatrix(r,kHistMapOutputHoriz);
   return r;
}

TH2 *TUnfoldDensity::GetEmatrixTotal
(const char *histogramName,const char *histogramTitle,
 const char *distributionName,const char *axisSteering,
 Bool_t useAxisBinning)
{
   // get total error including systematic,statistical,background,tau errors
   //   histogramName:  name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   distributionName: for complex binning schemes specify the name
   //                of the requested distribution within the TUnfoldBinning
   //                object
   //   axisSteering:
   //       "pattern1;pattern2;...;patternN"
   //       patternI = axis[mode]
   //       axis = name or *
   //       mode = C|U|O
   //        C: collapse axis into one bin
   //        U: discarde underflow bin
   //        O: discarde overflow bin
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the output histogram
   TUnfoldBinning const *binning=fConstOutputBins->FindNode(distributionName);
   Int_t *binMap=0;
   TH2 *r=binning->CreateErrorMatrixHistogram
      (histogramName,useAxisBinning,&binMap,histogramTitle,axisSteering);
   if(r) {
      TUnfoldSys::GetEmatrixTotal(r,binMap);
   }
   if(binMap) delete [] binMap;
   return r;
}

TH2 *TUnfoldDensity::GetL
(const char *histogramName,const char *histogramTitle,Bool_t useAxisBinning)
{
   // return the matrix of regularisation conditions in a histogram
   // input:
   //   histogramName: name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   //   useAxisBinning: if true, try to use the axis bin widths
   //                   on the x-axis of the output histogram
   if(fRegularisationConditions &&
      (fRegularisationConditions->GetEndBin()-
       fRegularisationConditions->GetStartBin()!= fL->GetNrows())) {
      Warning("GetL",
              "remove invalid scheme of regularisation conditions %d %d",
              fRegularisationConditions->GetEndBin(),fL->GetNrows());
      delete fRegularisationConditions;
      fRegularisationConditions=0;
   }
   if(!fRegularisationConditions) {
      fRegularisationConditions=new TUnfoldBinning("regularisation",fL->GetNrows());
      Warning("GetL","create flat regularisation conditions scheme");
   }
   TH2 *r=TUnfoldBinning::CreateHistogramOfMigrations
      (fConstOutputBins,fRegularisationConditions,histogramName,
       useAxisBinning,useAxisBinning,histogramTitle);
   TUnfold::GetL(r);
   return r;
}

TH1 *TUnfoldDensity::GetLxMinusBias
(const char *histogramName,const char *histogramTitle)
{
   // get regularisation conditions multiplied by result vector minus bias
   //   L(x-biasScale*biasVector)
   // this is a measure of the level of regulartisation required per
   // regularisation condition
   // if there are (negative or positive) spikes,
   // these regularisation conditions dominate
   // over the other regularisation conditions
   // input
   //   histogramName: name of the histogram
   //   histogramTitle: title of the histogram (could be zero)
   TMatrixD dx(*GetX(), TMatrixD::kMinus, fBiasScale * (*fX0));
   TMatrixDSparse *Ldx=MultiplyMSparseM(fL,&dx);
   if(fRegularisationConditions &&
      (fRegularisationConditions->GetEndBin()-
       fRegularisationConditions->GetStartBin()!= fL->GetNrows())) {
      Warning("GetLxMinusBias",
              "remove invalid scheme of regularisation conditions %d %d",
              fRegularisationConditions->GetEndBin(),fL->GetNrows());
      delete fRegularisationConditions;
      fRegularisationConditions=0;
   }
   if(!fRegularisationConditions) {
      fRegularisationConditions=new TUnfoldBinning("regularisation",fL->GetNrows());
      Warning("GetLxMinusBias","create flat regularisation conditions scheme");
   }
   TH1 *r=fRegularisationConditions->CreateHistogram
      (histogramName,kFALSE,0,histogramTitle);
   const Int_t *Ldx_rows=Ldx->GetRowIndexArray();
   const Double_t *Ldx_data=Ldx->GetMatrixArray();
   for(Int_t row=0;row<Ldx->GetNrows();row++) {
      if(Ldx_rows[row]<Ldx_rows[row+1]) {
         r->SetBinContent(row+1,Ldx_data[Ldx_rows[row]]);
      }
   }
   delete Ldx;
   return r;
}

const TUnfoldBinning *TUnfoldDensity::GetInputBinning
(const char *distributionName) const
{
   // find binning scheme, input bins
   //   distributionName : the distribution to locate
   return fConstInputBins->FindNode(distributionName);
}

const TUnfoldBinning *TUnfoldDensity::GetOutputBinning
(const char *distributionName) const
{
   // find binning scheme, output bins
   //   distributionName : the distribution to locate
   return fConstOutputBins->FindNode(distributionName);
}

Int_t TUnfoldDensity::ScanTau
(Int_t nPoint,Double_t tauMin,Double_t tauMax,TSpline **scanResult,
 Int_t mode,const char *distribution,const char *axisSteering,
 TGraph **lCurvePlot,TSpline **logTauXPlot,TSpline **logTauYPlot)
{
   // scan some variable as a function of tau and determine the minimum
   // input:
   //   nPoint: number of points to be scanned on the resulting curve
   //   tauMin: smallest tau value to study
   //   tauMax: largest tau value to study
   //     if (mauMin,tauMax) do not correspond to a valid tau range
   //     (e.g. tauMin=tauMax=0.0) then the tau range is determined
   //     automatically
   //   mode,distribution,axisSteering: argument to GetScanVariable()
   // output:
   //   scanResult: output spline of the variable as a function of tau
   // the following plots are produced on request (if pointers are non-zero)
   //   lCurvePlot: for monitoring: the L-curve
   //   logTauXPlot: for monitoring: L-curve(x) as a function of log(tau)
   //   logTauYPlot: for monitoring: L-curve(y) as a function of log(tau)
   // return value: the coordinate number (0..nPoint-1) corresponding to the
   //   final choice of tau
   typedef std::map<Double_t,Double_t> TauScan_t;
   typedef std::map<Double_t,std::pair<Double_t,Double_t> > LCurve_t;
   TauScan_t curve;
   LCurve_t lcurve;

   //==========================================================
   // algorithm:
   //  (1) do the unfolding for nPoint-1 points
   //      and store the results in the map
   //        curve
   //    (1a) store minimum and maximum tau to curve
   //    (1b) insert additional points, until nPoint-1 values
   //          have been calculated
   //
   //  (2) determine the best choice of tau
   //      do the unfolding for this point
   //      and store the result in
   //        curve
   //  (3) return the result in scanResult

   //==========================================================
   //  (1) do the unfolding for nPoint-1 points
   //      and store the results in
   //        curve
   //    (1a) store minimum and maximum tau to curve

   if((tauMin<=0)||(tauMax<=0.0)||(tauMin>=tauMax)) {
      // here no range is given, has to be determined automatically
      // the maximum tau is determined from the chi**2 values
      // observed from unfolding without regulatisation

      // first unfolding, without regularisation
      DoUnfold(0.0);

      // if the number of degrees of freedom is too small, create an error
      if(GetNdf()<=0) {
         Error("ScanTau","too few input bins, NDF<=0 %d",GetNdf());
      }
      Double_t X0=GetLcurveX();
      Double_t Y0=GetLcurveY();
      Double_t y0=GetScanVariable(mode,distribution,axisSteering);
      Info("ScanTau","logtau=-Infinity y=%lf X=%lf Y=%lf",y0,X0,Y0);
      {
         // unfolding guess maximum tau and store it
         Double_t logTau=
            0.5*(TMath::Log10(GetChi2A()+3.*TMath::Sqrt(GetNdf()+1.0))
                 -GetLcurveY());
         DoUnfold(TMath::Power(10.,logTau));
         if((!TMath::Finite(GetLcurveX())) ||(!TMath::Finite(GetLcurveY()))) {
            Fatal("ScanTau","problem (missing regularisation?) X=%f Y=%f",
                  GetLcurveX(),GetLcurveY());
         }
         Double_t y=GetScanVariable(mode,distribution,axisSteering);
         curve[logTau]=y;
         lcurve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
         Info("ScanTau","logtau=%lf y=%lf X=%lf Y=%lf",logTau,y,
              GetLcurveX(),GetLcurveY());
      }
      // minimum tau is chosen such that it is less than
      // 1% different from the case of no regularisation
      // here, several points are inserted as needed
      while(((int)curve.size()<nPoint-1)&&
            ((TMath::Abs(GetLcurveX()-X0)>0.00432)||
             (TMath::Abs(GetLcurveY()-Y0)>0.00432))) {
         Double_t logTau=(*curve.begin()).first-0.5;
         DoUnfold(TMath::Power(10.,logTau));
         Double_t y=GetScanVariable(mode,distribution,axisSteering);
         curve[logTau]=y;
         lcurve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
         Info("ScanTay","logtau=%lf y=%lf X=%lf Y=%lf",logTau,y,
              GetLcurveX(),GetLcurveY());
      }
   } else {
      Double_t logTauMin=TMath::Log10(tauMin);
      Double_t logTauMax=TMath::Log10(tauMax);
      if(nPoint>1) {
         // insert maximum tau
         DoUnfold(TMath::Power(10.,logTauMax));
         Double_t y=GetScanVariable(mode,distribution,axisSteering);
         curve[logTauMax]=y;
         lcurve[logTauMax]=std::make_pair(GetLcurveX(),GetLcurveY());
         Info("ScanTau","logtau=%lf y=%lf X=%lf Y=%lf",logTauMax,y,
              GetLcurveX(),GetLcurveY());
      }
      // insert minimum tau
      DoUnfold(TMath::Power(10.,logTauMin));
      Double_t y=GetScanVariable(mode,distribution,axisSteering);
      curve[logTauMin]=y;
      lcurve[logTauMin]=std::make_pair(GetLcurveX(),GetLcurveY());
      Info("ScanTau","logtau=%lf y=%lf X=%lf Y=%lf",logTauMin,y,
           GetLcurveX(),GetLcurveY());
   }

   //==========================================================
   //    (1b) insert additional points, until nPoint-1 values
   //          have been calculated
   while((int)curve.size()<nPoint-1) {
      // insert additional points
      // points are inserted such that the largest interval in log(tau)
      // is divided into two smaller intervals
      // however, there is a penalty term for subdividing intervals
      // which are far away from the minimum
      TauScan_t::const_iterator i0,i1;
      i0=curve.begin();
      // locate minimum
      Double_t logTauYMin=(*i0).first;
      Double_t yMin=(*i0).second;
      for(;i0!=curve.end();i0++) {
         if((*i0).second<yMin) {
            yMin=(*i0).second;
            logTauYMin=(*i0).first;
         }
      }
      // insert additional points such that large log(tau) intervals
      // near the minimum rho are divided into two
      i0=curve.begin();
      i1=i0;
      Double_t distMax=0.0;
      Double_t logTau=0.0;
      for(i1++;i1!=curve.end();i1++) {
         Double_t dist;
         // check size of rho interval
         dist=TMath::Abs((*i0).first-(*i1).first)
            // penalty term if distance from rhoMax is large
            +0.25*TMath::Power(0.5*((*i0).first+(*i1).first)-logTauYMin,2.)/
            ((*curve.rbegin()).first-(*curve.begin()).first)/nPoint;
         if((dist<=0.0)||(dist>distMax)) {
            distMax=dist;
            logTau=0.5*((*i0).first+(*i1).first);
         }
         i0=i1;
      }
      DoUnfold(TMath::Power(10.,logTau));
      Double_t y=GetScanVariable(mode,distribution,axisSteering);
      curve[logTau]=y;
      lcurve[logTau]=std::make_pair(GetLcurveX(),GetLcurveY());
      Info("ScanTau","logtau=%lf y=%lf X=%lf Y=%lf",logTau,y,
           GetLcurveX(),GetLcurveY());
   }

   //==========================================================
   //  (2) determine the best choice of tau
   //      do the unfolding for this point
   //      and store the result in
   //        curve

   Double_t cTmin=0.0;
   {
   Double_t *cTi=new Double_t[curve.size()];
   Double_t *cCi=new Double_t[curve.size()];
   Int_t n=0;
   for(TauScan_t::const_iterator i=curve.begin();i!=curve.end();i++) {
      cTi[n]=(*i).first;
      cCi[n]=(*i).second;
      n++;
   }
   // create rho Spline
   TSpline3 *splineC=new TSpline3("L curve curvature",cTi,cCi,n);
   // find the maximum of the curvature
   // if the parameter iskip is non-zero, then iskip points are
   // ignored when looking for the largest curvature
   // (there are problems with the curvature determined from the first
   //  few points of splineX,splineY in the algorithm above)
   Int_t iskip=0;
   if(n>3) iskip=1;
   if(n>6) iskip=2;
   Double_t cCmin=cCi[iskip];
   cTmin=cTi[iskip];
   for(Int_t i=iskip;i<n-1-iskip;i++) {
      // find minimum on this spline section
      // check boundary conditions for x[i+1]
      Double_t xMin=cTi[i+1];
      Double_t yMin=cCi[i+1];
      if(cCi[i]<yMin) {
         yMin=cCi[i];
         xMin=cTi[i];
      }
      // find minimum for x[i]<x<x[i+1]
      // get spline coefficiencts and solve equation
      //   derivative(x)==0
      Double_t x,y,b,c,d;
      splineC->GetCoeff(i,x,y,b,c,d);
      // coefficiencts of quadratic equation
      Double_t m_p_half=-c/(3.*d);
      Double_t q=b/(3.*d);
      Double_t discr=m_p_half*m_p_half-q;
      if(discr>=0.0) {
         // solution found
         discr=TMath::Sqrt(discr);
         Double_t xx;
         if(m_p_half>0.0) {
            xx = m_p_half + discr;
         } else {
            xx = m_p_half - discr;
         }
         Double_t dx=cTi[i+1]-x;
         // check first solution
         if((xx>0.0)&&(xx<dx)) {
            y=splineC->Eval(x+xx);
            if(y<yMin) {
               yMin=y;
               xMin=x+xx;
            }
         }
         // second solution
         if(xx !=0.0) {
            xx= q/xx;
         } else {
            xx=0.0;
         }
         // check second solution
         if((xx>0.0)&&(xx<dx)) {
            y=splineC->Eval(x+xx);
            if(y<yMin) {
               yMin=y;
               xMin=x+xx;
            }
         }
      }
      // check whether this local minimum is a global minimum
      if(yMin<cCmin) {
         cCmin=yMin;
         cTmin=xMin;
      }
   }
   delete splineC;
   delete[] cTi;
   delete[] cCi;
   }
   Double_t logTauFin=cTmin;
   DoUnfold(TMath::Power(10.,logTauFin));
   {
      Double_t y=GetScanVariable(mode,distribution,axisSteering);
      curve[logTauFin]=y;
      lcurve[logTauFin]=std::make_pair(GetLcurveX(),GetLcurveY());
      Info("ScanTau","Result logtau=%lf y=%lf X=%lf Y=%lf",logTauFin,y,
           GetLcurveX(),GetLcurveY());
   }
   //==========================================================
   //  (3) return the result in
   //       scanResult lCurve logTauX logTauY

   Int_t bestChoice=-1;
   if(curve.size()>0) {
      Double_t *y=new Double_t[curve.size()];
      Double_t *logT=new Double_t[curve.size()];
      int n=0;
      for( TauScan_t::const_iterator i=curve.begin();i!=curve.end();i++) {
         if(logTauFin==(*i).first) {
            bestChoice=n;
         }
         y[n]=(*i).second;
         logT[n]=(*i).first;
         n++;
      }
      if(scanResult) {
         TString name;
         name = TString::Format("scan(%d,",mode);
         if(distribution) name+= distribution;
         name += ",";
         if(axisSteering) name += axisSteering;
         name +=")";
         (*scanResult)=new TSpline3(name+"%log(tau)",logT,y,n);
      }
      delete[] y;
      delete[] logT;
   }
   if(lcurve.size()) {
      Double_t *logT=new Double_t[lcurve.size()];
      Double_t *x=new Double_t[lcurve.size()];
      Double_t *y=new Double_t[lcurve.size()];
      Int_t n=0;
      for(LCurve_t::const_iterator i=lcurve.begin();i!=lcurve.end();i++) {
         logT[n]=(*i).first;
         x[n]=(*i).second.first;
         y[n]=(*i).second.second;
         //cout<<logT[n]<<" "<< x[n]<<" "<<y[n]<<"\n";
         n++;
      }
      if(lCurvePlot) {
         *lCurvePlot=new TGraph(n,x,y);
         (*lCurvePlot)->SetTitle("L curve");
      }
      if(logTauXPlot)
         *logTauXPlot=new TSpline3("log(chi**2)%log(tau)",logT,x,n);
      if(logTauYPlot)
         *logTauYPlot=new TSpline3("log(reg.cond)%log(tau)",logT,y,n);
      delete [] y;
      delete [] x;
      delete [] logT;
   }
   return bestChoice;
}

Double_t TUnfoldDensity::GetScanVariable
(Int_t mode,const char *distribution,const char *axisSteering)
{
   // calculate variable for ScanTau()
   // the unfolding is repeated for various choices of tau.
   // For each tau, after unfolding, the ScanTau() method calls
   // GetScanVariable() to determine the value of the variable which
   // is to be scanned
   //
   // the variable is expected to have a minimum near the "optimal" choice
   // of tau
   //
   // input:
   //    mode : define the type of variable to be calculated
   //    distribution : define the distribution for which the variable
   //              is to be calculated
   //        the following modes are implemented:
   //          kEScanTauRhoAvg : average global correlation from input data
   //          kEScanTauRhoSquaredAvg : average global correlation squared
   //                                   from input data
   //          kEScanTauRhoMax : maximum global correlation from input data
   //          kEScanTauRhoAvgSys : average global correlation
   //                                 including systematic uncertainties
   //          kEScanTauRhoAvgSquaredSys : average global correlation squared
   //                                 including systematic uncertainties
   //          kEScanTauRhoMaxSys : maximum global correlation
   //                                 including systematic uncertainties
   //    distribution : name of the TUnfoldBinning node
   //                   for which to calculate the correlations
   //    axisSteering : axis steering for calculating the correlations
   //              the distribution
   // return: the value of the variable as determined from the present
   //    unfolding

   Double_t r=0.0;
   TString name("GetScanVariable(");
   name += TString::Format("%d,",mode);
   if(distribution) name += distribution;
   name += ",";
   if(axisSteering) name += axisSteering;
   name += ")";
   TH1 *rhoi=0;
   if((mode==kEScanTauRhoAvg)||(mode==kEScanTauRhoMax)||
      (mode==kEScanTauRhoSquareAvg)) {
      rhoi=GetRhoIstatbgr(name,0,distribution,axisSteering,kFALSE);
   } else if((mode==kEScanTauRhoAvgSys)||(mode==kEScanTauRhoMaxSys)||
             (mode==kEScanTauRhoSquareAvgSys)) {
      rhoi=GetRhoItotal(name,0,distribution,axisSteering,kFALSE);
   }
   if(rhoi) {
      Double_t sum=0.0;
      Double_t sumSquare=0.0;
      Double_t rhoMax=0.0;
      Int_t n=0;
      for(Int_t i=0;i<=rhoi->GetNbinsX()+1;i++) {
         Double_t c=rhoi->GetBinContent(i);
         if(c>=0.) {
            if(c>rhoMax) rhoMax=c;
            sum += c;
            sumSquare += c*c;
            n ++;
         }
      }
      if((mode==kEScanTauRhoAvg)||(mode==kEScanTauRhoAvgSys)) {
         r=sum/n;
      } else if((mode==kEScanTauRhoSquareAvg)||
                (mode==kEScanTauRhoSquareAvgSys)) {
         r=sum/n;
      } else {
         r=rhoMax;
      }
      // cout<<r<<" "<<GetRhoAvg()<<" "<<GetRhoMax()<<"\n";
      delete rhoi;
   } else {
      Fatal("GetScanVariable","mode %d not implemented",mode);
   }
   return r;
}
