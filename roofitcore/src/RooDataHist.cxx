/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataHist.cc,v 1.6 2001/10/04 01:44:33 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [DATA] --
// RooDataSet is a container class to hold N-dimensional binned data. Each bins central 
// coordinates in N-dimensional space are represented by a RooArgSet of RooRealVar, RooCategory 
// or RooStringVar objects, thus data can be binned in real and/or discrete dimensions
//

#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooAbsLValue.hh"


ClassImp(RooDataHist) 
;


RooDataHist::RooDataHist() 
{
  // Default constructor
  _arrSize = 0 ;
  _wgt = 0 ;
  _idxMult = 0 ;
  _curWeight = 0 ;
}



RooDataHist::RooDataHist(const char *name, const char *title, const RooArgSet& vars) : 
  RooTreeData(name,title,vars), _curWeight(0) 
{
  // Constructor of an empty data hist from a RooArgSet defining the dimensions
  // of the data space. The range and number of bins in each dimensions are taken
  // from getFitMin()getFitMax(),getFitBins() of each RooAbsArg representing that
  // dimension.
  //
  // For real dimensions, the fit range and number of bins can be set independently
  // of the plot range and number of bins, but it is advisable to keep the
  // ratio of the plot bin width and the fit bin width an integer value.
  // For category dimensions, the fit ranges always comprises all defined states
  // and each state is always has its individual bin
  //
  // To effective achive binning of real dimensions with variable bins sizes,
  // construct a RooThresholdCategory of the real dimension to be binned variably.
  // Set the thresholds at the desired bin boundaries, and construct the
  // data hist as function of the threshold category instead of the real variable.

   initialize() ;
}


RooDataHist::RooDataHist(const char *name, const char *title, const RooArgSet& vars, const RooAbsData& data) :
  RooTreeData(name,title,vars), _curWeight(0)
{
  // Constructor of a data hist from an existing data collection (binned or unbinned)
  // The RooArgSet 'vars' defines the dimensions of the histogram. 
  // The range and number of bins in each dimensions are taken
  // from getFitMin()getFitMax(),getFitBins() of each RooAbsArg representing that
  // dimension.
  //
  // For real dimensions, the fit range and number of bins can be set independently
  // of the plot range and number of bins, but it is advisable to keep the
  // ratio of the plot bin width and the fit bin width an integer value.
  // For category dimensions, the fit ranges always comprises all defined states
  // and each state is always has its individual bin
  //
  // To effective achive binning of real dimensions with variable bins sizes,
  // construct a RooThresholdCategory of the real dimension to be binned variably.
  // Set the thresholds at the desired bin boundaries, and construct the
  // data hist as function of the threshold category instead of the real variable.
  //
  // If the constructed data hist has less dimensions that in source data collection,
  // all missing dimensions will be projected.

  initialize() ;
  add(data) ;
}



void RooDataHist::initialize()
{
  // Initialization procedure: allocate weights array, calculate
  // multipliers needed for N-space to 1-dim array jump table,
  // and fill the internal tree with all bin center coordinates

  // Allocate coefficients array
  _idxMult = new Int_t[_vars.getSize()] ;

  _arrSize = 1 ;
  _iterator->Reset() ;
  RooAbsLValue* arg ;
  Int_t n(0), i ;
  while(arg=dynamic_cast<RooAbsLValue*>(_iterator->Next())) {
    
    // Calculate sub-index multipliers for master index
    for (i=0 ; i<n ; i++) {
      _idxMult[i] *= arg->numFitBins() ;
    }
    _idxMult[n++] = 1 ;

    // Calculate dimension of weight array
    _arrSize *= arg->numFitBins() ;
  }  

  // Allocate and initialize weight array 
  _wgt = new Double_t[_arrSize] ;
  for (i=0 ; i<_arrSize ; i++) _wgt[i] = 0 ;


  // Fill TTree with bin center coordinates
  // Calculate plot bins of components from master index
  Int_t ibin ;
  for (ibin=0 ; ibin<_arrSize ; ibin++) {
    _iterator->Reset() ;
    RooAbsLValue* arg ;
    Int_t i(0), idx(0), tmp(ibin) ;
    while(arg=dynamic_cast<RooAbsLValue*>(_iterator->Next())) {
      idx  = tmp / _idxMult[i] ;
      tmp -= idx*_idxMult[i++] ;
      RooAbsLValue* arglv = dynamic_cast<RooAbsLValue*>(arg) ;
      arglv->setFitBin(idx) ;
    }
    Fill() ;
  }
}



RooDataHist::RooDataHist(const RooDataHist& other, const char* newname) :
  RooTreeData(other,newname), _curWeight(0) 
{
  // Copy constructor

  Int_t i ;

  Int_t nVar = _vars.getSize() ;
  _idxMult = new Int_t[nVar] ;
  for (i=0 ; i<nVar ; i++) {
    _idxMult[i] = other._idxMult[i] ;  
  }

  // Allocate and initialize weight array 
  _arrSize = other._arrSize ;
  _wgt = new Double_t[_arrSize] ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = other._wgt[i] ;
  }  

}


RooDataHist::RooDataHist(const char* name, const char* title, RooDataHist* h, const RooArgSet& varSubset, 
			 const RooFormulaVar* cutVar, Bool_t copyCache) :
  RooTreeData(name,title,h,varSubset,cutVar, copyCache)
{
  // Constructor of a data hist from (part of) an existing data hist. The dimensions
  // of the data set are defined by the 'vars' RooArgSet, which can be identical
  // to 'dset' dimensions, or a subset thereof. Reduced dimensions will be projected
  // in the output data hist. The optional 'cutVar' formula variable can used to 
  // select the subset of bins to be copied.
  //
  // For most uses the RooAbsData::reduce() wrapper function, which uses this constructor, 
  // is the most convenient way to create a subset of an existing data
}



RooAbsData* RooDataHist::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache) 
{
  // Implementation of RooAbsData virtual method that drives the RooAbsData::reduce() methods
  RooDataHist *rdh = new RooDataHist(GetName(), GetTitle(), varSubset) ;
  rdh->add(*this,cutVar) ;
  return rdh ;
}



RooDataHist::~RooDataHist() 
{
  // Destructor

  if (_wgt) delete _wgt ;
  if (_idxMult) delete[] _idxMult ;
}


Int_t RooDataHist::calcTreeIndex() const 
{
  // Calculate the index for the weights array corresponding to 
  // to the bin enclosing the current coordinates of the internal argset

  _iterator->Reset() ;
  RooAbsLValue* arg ;
  Int_t masterIdx(0), i(0) ;
  while(arg=dynamic_cast<RooAbsLValue*>(_iterator->Next())) {
    masterIdx += _idxMult[i++]*arg->getFitBin() ;
  }
  return masterIdx ;
}


void RooDataHist::dump2() 
{
  // Debug stuff, should go...
  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    cout << "wgt[" << i << "] = " << _wgt[i] << endl ;
  }
}


Double_t RooDataHist::weight(const RooArgSet& bin) 
{
  // Return the weight for the bin enclosing the coordinates
  // given by the 'bin' argset.

  _vars = bin ;
 return _wgt[calcTreeIndex()] ;
}



void RooDataHist::add(const RooArgSet& row, Double_t weight) 
{
  // Increment the weight of the bin enclosing the coordinates
  // given by 'row' by the specified amount

  _vars = row ;
  _wgt[calcTreeIndex()] += weight ;
}


void RooDataHist::add(const RooAbsData& dset, const char* cut, Double_t weight) 
{  
  // Add all data points contained in 'dset' to this data set with given weight.
  // Optional cut string expression selects the data points to be added and can
  // reference any variable contained in this data set

  RooFormulaVar cutVar("select",cut,*dset.get()) ;
  add(dset,&cutVar,weight) ;
}



void RooDataHist::add(const RooAbsData& dset, const RooFormulaVar* cutVar, Double_t weight) 
{
  // Add all data points contained in 'dset' to this data set with given weight.
  // Optional RooFormulaVar pointer selects the data points to be added.

  RooFormulaVar* cloneVar(0) ;
  RooArgSet* tmp ;
  if (cutVar) {
    // Deep clone cutVar and attach clone to this dataset
    tmp = (RooArgSet*) RooArgSet(*cutVar).snapshot() ;
    cloneVar = (RooFormulaVar*) tmp->find(cutVar->GetName()) ;
    cloneVar->attachDataSet(dset) ;
    cloneVar->Print("v") ;
  }


  Int_t i ;
  for (i=0 ; i<dset.numEntries() ; i++) {
    const RooArgSet* row = dset.get(i) ;
    if (!cloneVar || cloneVar->getVal()) {
      add(*row,weight*dset.weight()) ;
    }
  }

  if (cloneVar) {
    delete tmp ;
  } 
}



Double_t RooDataHist::sum(Bool_t correctForBinSize) const 
{
  // Return the sum of the weights of all hist bins.
  //
  // If correctForBinSize is specified, the sum of weights
  // is multiplied by the N-dimensional bin volume,
  // making the return value the integral over the function
  // represented by this histogram

  Double_t binVolume(1) ;
  
  if (correctForBinSize) {
    _iterator->Reset() ;
    RooAbsLValue* arg ;
    while(arg=dynamic_cast<RooAbsLValue*>(_iterator->Next())) {
      if (!dynamic_cast<RooAbsReal*>(arg)) continue ;
      binVolume *= arg->getFitBinWidth() ;
    }
  }

  Int_t i ;
  Double_t total(0) ;
  for (i=0 ; i<_arrSize ; i++) {
    total += _wgt[i] ;
  }

  return total*binVolume ;
}




Double_t RooDataHist::sum(const RooArgSet& sumSet, const RooArgSet& sliceSet, Bool_t correctForBinSize)
{
  // Return the sum of the weights of a multi-dimensional slice of the histogram
  // by summing only over the dimensions specified in sumSet.
  //   
  // The coordinates of all other dimensions are fixed to those given in sliceSet
  //
  // If correctForBinSize is specified, the sum of weights
  // is multiplied by the M-dimensional bin volume, (M = N(sumSet)),
  // making the return value the integral over the function
  // represented by this histogram

  _vars = sliceSet ;

  Double_t binVolume(1) ;
  TIterator* ssIter = sumSet.createIterator() ;
  
  // Calculate binVolume if correction for that is requested
  RooAbsLValue* rArg ;
  if (correctForBinSize) {
    while(rArg=dynamic_cast<RooAbsLValue*>(ssIter->Next())) {
      if (!dynamic_cast<RooAbsReal*>(rArg)) continue ;
      binVolume *= rArg->getFitBinWidth() ;
    }
  }
  
  // Calculate mask and refence plot bins for non-iterating variables
  RooAbsArg* arg ;
  Bool_t* mask = new Bool_t[_vars.getSize()] ;
  Int_t*  refBin = new Int_t[_vars.getSize()] ;
  Int_t i(0) ;
  _iterator->Reset() ;
  while(arg=(RooAbsArg*)_iterator->Next()) {
    if (sumSet.find(arg->GetName())) {
      mask[i] = kFALSE ;
    } else {
      mask[i] = kTRUE ;
      refBin[i] = (dynamic_cast<RooAbsLValue*>(arg))->getFitBin() ;
    }
    i++ ;
  }
    
  // Loop over entire data set, skipping masked entries
  Double_t total(0) ;
  Int_t ibin ;
  for (ibin=0 ; ibin<_arrSize ; ibin++) {

    Int_t idx(0), tmp(ibin), ivar(0) ;
    Bool_t skip(kFALSE) ;

    // Check if this bin belongs in selected slice
    _iterator->Reset() ;
    while(!skip && (arg=(RooAbsArg*)_iterator->Next())) {
      idx  = tmp / _idxMult[ivar] ;
      tmp -= idx*_idxMult[ivar] ;
      if (mask[ivar] && idx!=refBin[ivar]) skip=kTRUE ;
      ivar++ ;
    }
    
    if (!skip) total += _wgt[ibin] ;
  }
  delete ssIter ;

  return total*binVolume ;
}




Int_t RooDataHist::numEntries(Bool_t useWeights) const 
{
  // Return the number of bins (useWeights=false) or
  // the sum of the weights of all bins (useWeight=true)

  if (!useWeights) return RooTreeData::numEntries() ;

  Int_t i ;
  Double_t n(0) ;
  for (i=0 ; i<_arrSize ; i++) {
    n+= _wgt[i] ;
  }
  return Int_t(n) ;
}


void RooDataHist::reset() 
{
  // Reset all bin weights to zero

  RooTreeData::reset() ;

  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = 0. ;
  }
  _curWeight = 0 ;

}


const RooArgSet* RooDataHist::get(Int_t masterIdx) const  
{
  // Return an argset with the bin center coordinates for 
  // bin sequential number 'masterIdx'. For iterative use.

  _curWeight = _wgt[masterIdx] ;
  return RooTreeData::get(masterIdx) ;  
}





