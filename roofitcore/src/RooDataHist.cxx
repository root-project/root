/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataHist.cc,v 1.4 2001/09/27 18:22:29 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooAbsLValue.hh"


ClassImp(RooDataHist) 
;


RooDataHist::RooDataHist() 
{
  _arrSize = 0 ;
  _wgt = 0 ;
  _idxMult = 0 ;
  _curWeight = 0 ;
}



RooDataHist::RooDataHist(const char *name, const char *title, const RooArgSet& vars) : 
  RooTreeData(name,title,vars), _curWeight(0) 
{
  initialize() ;
}


RooDataHist::RooDataHist(const char *name, const char *title, const RooArgSet& vars, const RooAbsData& data) :
  RooTreeData(name,title,vars), _curWeight(0)
{
  initialize() ;
  add(data) ;
}



void RooDataHist::initialize()
{
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
}



RooAbsData* RooDataHist::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache) 
{
  RooDataHist *rdh = new RooDataHist(GetName(), GetTitle(), varSubset) ;
  rdh->add(*this,cutVar) ;
  return rdh ;
}



RooDataHist::~RooDataHist() 
{
  if (_wgt) delete _wgt ;
  if (_idxMult) delete[] _idxMult ;
}


Int_t RooDataHist::calcTreeIndex() const {
  // Calculate the master index corresponding to the current set of values in _var
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
  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    cout << "wgt[" << i << "] = " << _wgt[i] << endl ;
  }
}


Double_t RooDataHist::weight(const RooArgSet& bin) 
{
  _vars = bin ;
 return _wgt[calcTreeIndex()] ;
}



void RooDataHist::add(const RooArgSet& row, Double_t weight) 
{
  _vars = row ;
  _wgt[calcTreeIndex()] += weight ;
}


void RooDataHist::add(const RooAbsData& dset, const char* cut, Double_t weight) 
{  
  RooFormulaVar cutVar("select",cut,*dset.get()) ;
  add(dset,&cutVar,weight) ;
}



void RooDataHist::add(const RooAbsData& dset, const RooFormulaVar* cutVar, Double_t weight) 
{
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
  RooTreeData::reset() ;

  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = 0. ;
  }
  _curWeight = 0 ;

}


const RooArgSet* RooDataHist::get(Int_t masterIdx) const  
{
  _curWeight = _wgt[masterIdx] ;
  return RooTreeData::get(masterIdx) ;  
}





