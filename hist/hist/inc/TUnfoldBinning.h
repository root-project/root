// Author: Stefan Schmitt
// DESY, 10/08/11

//  Version 17.1, in parallel to TUnfold
//
//  History:
//    Version 17.0, initial version, numbered in parallel to TUnfold

#ifndef ROOT_TUnfoldBinning
#define ROOT_TUnfoldBinning


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//                                                                      //
//  TUnfoldBinning, an auxillary class to provide                       //
//  complex binning schemes as input to TUnfoldDensity                  //
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

#include "TUnfold.h"
#include <iostream>
#include <TNamed.h>
#include <TObjArray.h>
#include <TObjString.h>

class TAxis;
class TF1;


class TUnfoldBinning : public TNamed {
 protected:
   TUnfoldBinning *parentNode; // mother node
   TUnfoldBinning *childNode; // first daughter node
   TUnfoldBinning *nextNode; // next sister
   TUnfoldBinning *prevNode; // previous sister
   TObjArray *fAxisList; // for each axis the bin borders (TVectorD)
   TObjArray *fAxisLabelList; // for each axis its name (TObjString)
   Int_t fHasUnderflow,fHasOverflow; // bit fields indicating whether there are underflow/overflow bins on the axes
   Int_t fDistributionSize; // number of bins in this node's distribution
   Int_t fFirstBin; // global bin number of the first bin
   Int_t fLastBin; // global bin number of the last(+1) bin
   TF1 *fBinFactorFunction; // function to calculate user factor from bin centres (default function is a constant)
   Double_t fBinFactorConstant; // scale factor on user factor
 public:
   /********************* setup **************************/

   TUnfoldBinning(const char *name=0,Int_t nBins=0,const char *binNames=0); // create a new root node with a given number of unconnected bins
   TUnfoldBinning(const TAxis &axis,Int_t includeUnderflow,Int_t includeOverflow); // create a binning scheme with one axis
   TUnfoldBinning *AddBinning(TUnfoldBinning *binning); // add a new node to the TUnfoldBinning tree
   TUnfoldBinning *AddBinning(const char *name,Int_t nBins=0,const char *binNames=0); // add a new node to the TUnfoldBinning tree
   Bool_t AddAxis(const char *name,Int_t nBins,const Double_t *binBorders,
                Bool_t hasUnderflow,Bool_t hasOverflow); // add an axis (variable bins) to the distribution associated with this node
   Bool_t AddAxis(const char *name,Int_t nBins,Double_t xMin,Double_t xMax,
                Bool_t hasUnderflow,Bool_t hasOverflow); // add an axis (equidistant bins) to the distribution associated with this node
   Bool_t AddAxis(const TAxis &axis,Bool_t includeUnderflow,Bool_t includeOverflow); // add an axis (from TAxis instance) to the distribution associated with this node
   virtual ~TUnfoldBinning(void);
   void PrintStream(std::ostream &out,Int_t indent=0) const;
   inline void SetBinFactorFunction(Double_t normalisation,TF1 *userFunc=0) {
      fBinFactorConstant=normalisation; fBinFactorFunction=userFunc; }// define function to calculate bin factor

   /********************* Navigation **********************/
   inline TUnfoldBinning const *GetChildNode(void) const { return childNode; } // first daughter
   inline TUnfoldBinning const *GetPrevNode(void) const { return prevNode; } // previoous sister
   inline TUnfoldBinning const *GetNextNode(void) const { return nextNode; } // next sister
   inline TUnfoldBinning const *GetParentNode(void) const { return parentNode; } // mother
   TUnfoldBinning const *FindNode(char const *name) const; // find node by name

   /********************* Create THxx histograms **********/
   Int_t GetTH1xNumberOfBins(Bool_t originalAxisBinning=kTRUE,const char *axisSteering=0) const; // get number of bins of a one-dimensional histogram TH1
   TH1 *CreateHistogram(const char *histogramName,Bool_t originalAxisBinning=kFALSE,Int_t **binMap=0,const char *histogramTitle=0,const char *axisSteering=0) const; // create histogram and bin map for this node
   TH2D *CreateErrorMatrixHistogram(const char *histogramName,Bool_t originalAxisBinning,Int_t **binMap=0,const char *histogramTitle=0,const char *axisSteering=0) const; // create histogram and bin map for this node
   static TH2D *CreateHistogramOfMigrations(TUnfoldBinning const *xAxis,
                                           TUnfoldBinning const *yAxis,
                                           char const *histogramName,
                                           Bool_t originalXAxisBinning=kFALSE,
                                           Bool_t originalYAxisBinning=kFALSE,
                                           char const *histogramTitle=0); // create 2D histogram with one binning on the x axis and the other binning on the y axis
   TH1 *ExtractHistogram(const char *histogramName,const TH1 *globalBins,const TH2 *globalBinsEmatrix=0,Bool_t originalAxisBinning=kTRUE,const char *axisSteering=0) const; // extract a distribution from the given set of global bins
   /********************* Calculate global bin number ******/
   Int_t GetGlobalBinNumber(Double_t x) const; // get bin number 1-dim distribution
   Int_t GetGlobalBinNumber(Double_t x,Double_t y) const; // get bin number 2-dim distribution
   Int_t GetGlobalBinNumber(Double_t x,Double_t y,Double_t z) const; // get bin number 3-dim distribution
   Int_t GetGlobalBinNumber(Double_t x0,Double_t x1,Double_t x2,Double_t x3) const; // get bin number for given variables, up to four-dimensional binning
   Int_t GetGlobalBinNumber(const Double_t *x) const; // get bin number, up to 32 dimenstional binning
   inline Int_t GetStartBin(void) const { return fFirstBin; } // first bin of this node
   inline Int_t GetEndBin(void) const { return fLastBin; } // last+1 bin of this node (includes children)

    /********************* access by global bin number ******/
   TString GetBinName(Int_t iBin) const; // return bin name
   Double_t GetBinSize(Int_t iBin) const; // return bin size (in N dimensions)
   virtual Double_t GetBinFactor(Int_t iBin) const; // return user factor
   void GetBinUnderflowOverflowStatus(Int_t iBin,Int_t *uStatus,Int_t *oStatus) const; // return bit map indicating underflow and overflow status
   void GetBinNeighbours(Int_t globalBin,Int_t axis,
                         Int_t *prev,Double_t *distPrev,
                         Int_t *next,Double_t *distNext) const; // get neighbour bins along an axis
   /********************* access by bin number, given an axis steering ******/
   enum { MAXDIM=32 };
   const TUnfoldBinning *GetBinLocation(Int_t binTHxx,const char *axisSteering,
                                  Int_t axisBins[MAXDIM]) const; //  locate a given THxx bin for a given axis steering
   void DecodeAxisSteering(const char *axisSteering,const char *options,
                           Int_t *isOptionGiven) const; // decode axis steering options
   /********************** access distribution properties *************/
   inline Int_t GetDistributionNumberOfBins(void) const { return fDistributionSize; } // number of bins in the distribution possibly including under/overflow
   inline Int_t GetDistributionDimension(void) const {  return fAxisList->GetEntriesFast(); } // query dimension of this node's distribution
   virtual Double_t GetDistributionAverageBinSize(Int_t axis,Bool_t includeUnderflow, Bool_t includeOverflow) const; // get average bin size
   inline TVectorD const *GetDistributionBinning(Int_t axis) const {
      return (TVectorD const *)fAxisList->At(axis); } // get bin borders for some axis
   inline TString GetDistributionAxisLabel(Int_t axis) const {
      return ((TObjString * const)fAxisLabelList->At(axis))->GetString(); }// get name of this axis

   virtual Double_t GetDistributionUnderflowBinWidth(Int_t axis) const; // width of underflow bin on the given axis
   virtual Double_t GetDistributionOverflowBinWidth(Int_t axis) const; // width of overflow bin on the given axis
   virtual Double_t GetDistributionBinCenter(Int_t axis,Int_t bin) const; // position of bin center on the given axis
 protected:
   TUnfoldBinning *GetRootNode(void); // return root node
   TUnfoldBinning const *GetRootNode(void) const; // return root node
   void Initialize(Int_t nBins);
   Int_t UpdateFirstLastBin(Bool_t startWithRootNode=kTRUE); // update fFirstBin and fLastBin
   Bool_t HasUnconnectedBins(void) const; // check whether this node has bins without axis
   TUnfoldBinning const *ToAxisBins(Int_t globalBin,Int_t *axisBins) const; // return distribution in which the bin is located
   Int_t ToGlobalBin(Int_t const *axisBins) const; // return -1 if not inside distribution
   TString BuildHistogramTitle(const char *histogramName,const char *histogramTitle,
                               Int_t const *axisList) const; // construct histogram title
   TString BuildHistogramTitle2D(const char *histogramName,const char *histogramTitle,
                                 Int_t xAxis,const TUnfoldBinning *yAxisBinning,Int_t yAxis) const; // construct histogram title
   Int_t GetTHxxBinning(Int_t maxDim,Int_t *axisBins,Int_t *axisList,const char *axisSteering) const; // get binning information for creating a THxx
   Int_t GetTHxxBinningSingleNode(Int_t maxDim,Int_t *axisBins,Int_t *axisList,const char *axisSteering) const; // get binning information for creating a THxx
   Int_t GetTHxxBinsRecursive(const char *axisSteering) const; // get binning information for creating a THxx
   const TUnfoldBinning *GetBinLocationRecursive(Int_t &offset,const char *axisSteering,
                                  Int_t axisBins[MAXDIM]) const; //  locate a THxx bin offset for a given axis steering
   const TUnfoldBinning *GetNonemptyNode(void) const; // get the only nodes with non-empty distributions if there are multiple nodes, return 0
   Int_t *CreateBinMap(const TH1 *hist,Int_t nDim,const Int_t *axisList,const char *axisSteering) const; // create mapping from global bins to a histogram
   Int_t FillBinMapRecursive(Int_t startBin,const char *axisSteering,
                            Int_t *binMap) const; // fill bin map recursively
   Int_t FillBinMapSingleNode(const TH1 *hist,Int_t startBin,Int_t nDim,const Int_t *axisList,const char *axisSteering,Int_t *binMap) const; // fill bin map for a single node

   ClassDef(TUnfoldBinning, TUnfold_CLASS_VERSION) //Complex binning schemes for TUnfoldDensity
};

#endif
