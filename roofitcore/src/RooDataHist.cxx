/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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
  // Allocate coefficients array
  _idxMult = new Int_t[vars.getSize()] ;

  _arrSize = 1 ;
  _iterator->Reset() ;
  RooAbsArg* arg ;
  Int_t n(0), i ;
  while(arg=(RooAbsArg*)_iterator->Next()) {
    
    // Calculate sub-index multipliers for master index
    for (i=0 ; i<n ; i++) {
      _idxMult[i] *= arg->numPlotBins() ;
    }
    _idxMult[n++] = 1 ;

    // Calculate dimension of weight array
    _arrSize *= arg->numPlotBins() ;
  }  

  // Allocate and initialize weight array 
  _wgt = new Double_t[_arrSize] ;


  // Fill TTree with bin center coordinates
  // Calculate plot bins of components from master index
  Int_t ibin ;
  for (ibin=0 ; ibin<_arrSize ; ibin++) {
    _iterator->Reset() ;
    RooAbsArg* arg ;
    Int_t i(0), idx(0), tmp(ibin) ;
    while(arg=(RooAbsArg*)_iterator->Next()) {
      idx  = tmp / _idxMult[i] ;
      tmp -= idx*_idxMult[i++] ;
      RooAbsLValue* arglv = dynamic_cast<RooAbsLValue*>(arg) ;
      arglv->setPlotBin(idx) ;
    }
    Fill() ;
  }
}


RooDataHist::RooDataHist(const RooDataHist& other, const char* newname) :
  RooTreeData(other,newname), _curWeight(0) 
{
}


RooDataHist::RooDataHist(const char* name, const char* title, RooDataHist* h, const RooArgSet& varSubset, 
			 const RooFormulaVar* cutVar, Bool_t copyCache) :
  RooTreeData(name,title,h,varSubset,cutVar, copyCache)
{
  // Need to collapse weight matrix here
  assert(0) ;
}



RooAbsData* RooDataHist::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache) 
{
  return new RooDataHist(GetName(), GetTitle(), this, varSubset, cutVar, copyCache) ;
}



RooDataHist::~RooDataHist() 
{
  if (_wgt) delete _wgt ;
  if (_idxMult) delete[] _idxMult ;
}


Int_t RooDataHist::calcTreeIndex() const {
  // Calculate the master index corresponding to the current set of values in _var
  _iterator->Reset() ;
  RooAbsArg* arg ;
  Int_t masterIdx(0), i(0) ;
  while(arg=(RooAbsArg*)_iterator->Next()) {
    masterIdx += _idxMult[i++]*arg->getPlotBin() ;
  }
  return masterIdx ;
}


void RooDataHist::add(const RooArgSet& row, Double_t weight) 
{
  _vars = row ;
  _wgt[calcTreeIndex()] += weight ;
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


Roo1DTable* RooDataHist::table(RooAbsCategory& cat, const char* cuts, const char* opts) const
{
  return 0 ;
}


RooPlot* RooDataHist::plotOn(RooPlot *frame, const char* cuts, Option_t* drawOptions) const 
{
  return 0 ;
}


