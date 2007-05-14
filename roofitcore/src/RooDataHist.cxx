/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooDataHist.cxx,v 1.52 2007/05/11 09:11:58 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [DATA] --
// RooDataSet is a container class to hold N-dimensional binned data. Each bins central 
// coordinates in N-dimensional space are represented by a RooArgSet of RooRealVar, RooCategory 
// or RooStringVar objects, thus data can be binned in real and/or discrete dimensions
//

#include "RooFit.h"

#include "TH1.h"
#include "TH1.h"
#include "TMath.h"
#include "RooDataHist.h"
#include "RooAbsLValue.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooMath.h"
#include "RooBinning.h"
#include "RooPlot.h"
#include "RooHistError.h"

ClassImp(RooDataHist) 
;


RooDataHist::RooDataHist() 
{
  // Default constructor
  _arrSize = 0 ;
  _wgt = 0 ;
  _errLo = 0 ;
  _errHi = 0 ;
  _sumw2 = 0 ;
  _binv = 0 ;
  _pbinv = 0 ;
  _idxMult = 0 ;  
  _curWeight = 0 ;
  _curIndex = -1 ;
  _realIter = 0 ;

}



RooDataHist::RooDataHist(const char *name, const char *title, const RooArgSet& vars) : 
  RooTreeData(name,title,vars), _curWeight(0), _curVolume(1), _pbinv(0)
{
  // Constructor of an empty data hist from a RooArgSet defining the dimensions
  // of the data space. The range and number of bins in each dimensions are taken
  // from getMin()getMax(),getBins() of each RooAbsArg representing that
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
  appendToDir(this,kTRUE) ;
}


RooDataHist::RooDataHist(const char *name, const char *title, const RooArgSet& vars, const RooAbsData& data, Double_t weight) :
  RooTreeData(name,title,vars), _curWeight(0), _curVolume(1), _pbinv(0)
{
  // Constructor of a data hist from an existing data collection (binned or unbinned)
  // The RooArgSet 'vars' defines the dimensions of the histogram. 
  // The range and number of bins in each dimensions are taken
  // from getMin()getMax(),getBins() of each RooAbsArg representing that
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
  add(data,(const RooFormulaVar*)0,weight) ;
  appendToDir(this,kTRUE) ;
}


RooDataHist::RooDataHist(const char *name, const char *title, const RooArgList& vars, const TH1* hist, Double_t weight) :
  RooTreeData(name,title,vars), _curWeight(0), _curVolume(1), _pbinv(0)
{
  // Constructor of a data hist from an TH1,TH2 or TH3
  // The RooArgSet 'vars' defines the dimensions of the histogram. The ranges
  // and number of bins are taken from the input histogram, and the corresponding
  // values are set accordingly on the arguments in 'vars'

  // Check consistency in number of dimensions
  if (vars.getSize() != hist->GetDimension()) {
    cout << "RooDataHist::ctor(" << GetName() << ") ERROR: dimension of input histogram must match "
	 << "number of dimension variables" << endl ;
    assert(0) ; 
  }

  // Copy fitting and plotting bins/ranges from TH1 to dimension variables
  // Int_t nDim = vars.getSize() ;
  TH1* histo = const_cast<TH1*>(hist) ;

  // X
  RooRealVar* xvar = (RooRealVar*) _vars.find(vars.at(0)->GetName()) ;
  if (!dynamic_cast<RooRealVar*>(xvar)) {
    cout << "RooDataHist::ctor(" << GetName() << ") ERROR: dimension " << xvar->GetName() << " must be real" << endl ;
    assert(0) ;
  }

  Double_t xlo = ((RooRealVar*)vars.at(0))->getMin() ;
  Double_t xhi = ((RooRealVar*)vars.at(0))->getMax() ;
  Int_t xmin(0) ;
  if (histo->GetXaxis()->GetXbins()->GetArray()) {

    RooBinning xbins(histo->GetNbinsX(),histo->GetXaxis()->GetXbins()->GetArray()) ;

    // Adjust xlo/xhi to nearest boundary
    Double_t xloAdj = xbins.binLow(xbins.binNumber(xlo+1e-6)) ;
    Double_t xhiAdj = xbins.binHigh(xbins.binNumber(xhi-1e-6)) ;
    xbins.setRange(xloAdj,xhiAdj) ;
    ((RooRealVar*)vars.at(0))->setBinning(xbins) ;
    if (fabs(xloAdj-xlo)>1e-6||fabs(xhiAdj-xhi)) {
      cout << "RooDataHist::ctor(" << GetName() << "): fit range of variable " << xvar->GetName() << " expanded to nearest bin boundaries: [" 
	   << xlo << "," << xhi << "] --> [" << xloAdj << "," << xhiAdj << "]" << endl ;
    }

    xvar->setBinning(xbins) ;
    xmin = xbins.rawBinNumber(xloAdj+1e-6) ;

  } else {
    RooBinning xbins(histo->GetXaxis()->GetXmin(),histo->GetXaxis()->GetXmax()) ;
    xbins.addUniform(histo->GetNbinsX(),histo->GetXaxis()->GetXmin(),histo->GetXaxis()->GetXmax()) ;

    // Adjust xlo/xhi to nearest boundary
    Double_t xloAdj = xbins.binLow(xbins.binNumber(xlo+1e-6)) ;
    Double_t xhiAdj = xbins.binHigh(xbins.binNumber(xhi-1e-6)) ;
    xbins.setRange(xloAdj,xhiAdj) ;
    ((RooRealVar*)vars.at(0))->setRange(xloAdj,xhiAdj) ;
    if (fabs(xloAdj-xlo)>1e-6||fabs(xhiAdj-xhi)) {
      cout << "RooDataHist::ctor(" << GetName() << "): fit range of variable " << xvar->GetName() << " expanded to nearest bin boundaries: [" 
	   << xlo << "," << xhi << "] --> [" << xloAdj << "," << xhiAdj << "]" << endl ;
    }

    RooUniformBinning xbins2(xloAdj,xhiAdj,xbins.numBins()) ;
    xvar->setBinning(xbins2) ;
    xmin = xbins.rawBinNumber(xloAdj+1e-6) ;
  }



  // Y
  RooRealVar* yvar = (RooRealVar*) (vars.at(1) ? _vars.find(vars.at(1)->GetName()) : 0 ) ;
  Int_t ymin(0) ;
  if (yvar) {
    Double_t ylo = ((RooRealVar*)vars.at(1))->getMin() ;
    Double_t yhi = ((RooRealVar*)vars.at(1))->getMax() ;

    if (!dynamic_cast<RooRealVar*>(yvar)) {
      cout << "RooDataHist::ctor(" << GetName() << ") ERROR: dimension " << yvar->GetName() << " must be real" << endl ;
      assert(0) ;
    }

    if (histo->GetYaxis()->GetXbins()->GetArray()) {

      RooBinning ybins(histo->GetNbinsY(),histo->GetYaxis()->GetXbins()->GetArray()) ;
      
      // Adjust ylo/yhi to nearest boundary
      Double_t yloAdj = ybins.binLow(ybins.binNumber(ylo+1e-6)) ;
      Double_t yhiAdj = ybins.binHigh(ybins.binNumber(yhi-1e-6)) ;
      ybins.setRange(yloAdj,yhiAdj) ;
      ((RooRealVar*)vars.at(1))->setBinning(ybins) ;
      if (fabs(yloAdj-ylo)>1e-6||fabs(yhiAdj-yhi)) {
	cout << "RooDataHist::ctor(" << GetName() << "): fit range of variable " << yvar->GetName() << " expanded to nearest bin boundaries: [" 
	     << ylo << "," << yhi << "] --> [" << yloAdj << "," << yhiAdj << "]" << endl ;
      }

      yvar->setBinning(ybins) ;
      ymin = ybins.rawBinNumber(yloAdj+1e-6) ;

    } else {

      RooBinning ybins(histo->GetYaxis()->GetXmin(),histo->GetYaxis()->GetXmax()) ;
      ybins.addUniform(histo->GetNbinsY(),histo->GetYaxis()->GetXmin(),histo->GetYaxis()->GetXmax()) ;
      
      // Adjust ylo/yhi to nearest boundary
      Double_t yloAdj = ybins.binLow(ybins.binNumber(ylo+1e-6)) ;
      Double_t yhiAdj = ybins.binHigh(ybins.binNumber(yhi-1e-6)) ;
      ybins.setRange(yloAdj,yhiAdj) ;
      ((RooRealVar*)vars.at(1))->setRange(yloAdj,yhiAdj) ;
      if (fabs(yloAdj-ylo)>1e-6||fabs(yhiAdj-yhi)) {
	cout << "RooDataHist::ctor(" << GetName() << "): fit range of variable " << yvar->GetName() << " expanded to nearest bin boundaries: [" 
	     << ylo << "," << yhi << "] --> [" << yloAdj << "," << yhiAdj << "]" << endl ;
      }
      
      RooUniformBinning ybins2(yloAdj,yhiAdj,ybins.numBins()) ;
      yvar->setBinning(ybins2) ;
      ymin = ybins.rawBinNumber(yloAdj+1e-6) ;
    }    
  }
  
  // Z
  RooRealVar* zvar = (RooRealVar*) (vars.at(2) ? _vars.find(vars.at(2)->GetName()) : 0 ) ;
  Int_t zmin(0) ;
  if (zvar) {
    Double_t zlo = ((RooRealVar*)vars.at(2))->getMin() ;
    Double_t zhi = ((RooRealVar*)vars.at(2))->getMax() ;

    if (!dynamic_cast<RooRealVar*>(zvar)) {
      cout << "RooDataHist::ctor(" << GetName() << ") ERROR: dimension " << zvar->GetName() << " must be real" << endl ;
      assert(0) ;
    }

    if (histo->GetZaxis()->GetXbins()->GetArray()) {

      RooBinning zbins(histo->GetNbinsZ(),histo->GetZaxis()->GetXbins()->GetArray()) ;
      
      // Adjust zlo/zhi to nearest boundary
      Double_t zloAdj = zbins.binLow(zbins.binNumber(zlo+1e-6)) ;
      Double_t zhiAdj = zbins.binHigh(zbins.binNumber(zhi-1e-6)) ;
      zbins.setRange(zloAdj,zhiAdj) ;
      ((RooRealVar*)vars.at(2))->setBinning(zbins) ;
      if (fabs(zloAdj-zlo)>1e-6||fabs(zhiAdj-zhi)) {
	cout << "RooDataHist::ctor(" << GetName() << "): fit range of variable " << zvar->GetName() << " expanded to nearest bin boundaries: [" 
	     << zlo << "," << zhi << "] --> [" << zloAdj << "," << zhiAdj << "]" << endl ;
      }
      
      zvar->setBinning(zbins) ;
      zmin = zbins.rawBinNumber(zloAdj+1e-6) ;
      
    } else {

      RooBinning zbins(histo->GetZaxis()->GetXmin(),histo->GetZaxis()->GetXmax()) ;
      zbins.addUniform(histo->GetNbinsZ(),histo->GetZaxis()->GetXmin(),histo->GetZaxis()->GetXmax()) ;
      
      // Adjust zlo/zhi to nearest boundary
      Double_t zloAdj = zbins.binLow(zbins.binNumber(zlo+1e-6)) ;
      Double_t zhiAdj = zbins.binHigh(zbins.binNumber(zhi-1e-6)) ;
      zbins.setRange(zloAdj,zhiAdj) ;
      ((RooRealVar*)vars.at(2))->setRange(zloAdj,zhiAdj) ;
      if (fabs(zloAdj-zlo)>1e-6||fabs(zhiAdj-zhi)) {
	cout << "RooDataHist::ctor(" << GetName() << "): fit range of variable " << zvar->GetName() << " expanded to nearest bin boundaries: [" 
	     << zlo << "," << zhi << "] --> [" << zloAdj << "," << zhiAdj << "]" << endl ;
      }
      
      RooUniformBinning zbins2(zloAdj,zhiAdj,zbins.numBins()) ;
      zvar->setBinning(zbins2) ;
      zmin = zbins.rawBinNumber(zloAdj+1e-6) ;
    }
  }
  

  // Initialize internal data structure
  initialize() ;
  appendToDir(this,kTRUE) ;

  // Transfer contents
  RooArgSet set(*xvar) ;
  if (yvar) set.add(*yvar) ;
  if (zvar) set.add(*zvar) ;

  Int_t ix(0),iy(0),iz(0) ;
  for (ix=0 ; ix < xvar->getBins() ; ix++) {
    xvar->setBin(ix) ;
    if (yvar) {
      for (iy=0 ; iy < yvar->getBins() ; iy++) {
	yvar->setBin(iy) ;
	if (zvar) {
	  for (iz=0 ; iz < zvar->getBins() ; iz++) {
	    zvar->setBin(iz) ;
	    add(set,histo->GetBinContent(ix+1+xmin,iy+1+ymin,iz+1+zmin)*weight,TMath::Power(histo->GetBinError(ix+1+xmin,iy+1+ymin,iz+1+zmin)*weight,2)) ;
	  }
	} else {
	  add(set,histo->GetBinContent(ix+1+xmin,iy+1+ymin)*weight,TMath::Power(histo->GetBinError(ix+1+xmin,iy+1+ymin)*weight,2)) ;
	}
      }
    } else {
      add(set,histo->GetBinContent(ix+1+xmin)*weight,TMath::Power(histo->GetBinError(ix+1+xmin)*weight,2)) ;	    
    }
  }  
  
}



void RooDataHist::initialize(Bool_t fillTree)
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
  while((arg=dynamic_cast<RooAbsLValue*>(_iterator->Next()))) {
    
    // Calculate sub-index multipliers for master index
    for (i=0 ; i<n ; i++) {
      _idxMult[i] *= arg->numBins() ;
    }
    _idxMult[n++] = 1 ;

    // Calculate dimension of weight array
    _arrSize *= arg->numBins() ;
  }  

  // Allocate and initialize weight array 
  _wgt = new Double_t[_arrSize] ;
  _errLo = new Double_t[_arrSize] ;
  _errHi = new Double_t[_arrSize] ;
  _sumw2 = new Double_t[_arrSize] ;
  _binv = new Double_t[_arrSize] ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = 0 ;
    _errLo[i] = -1 ;
    _errHi[i] = -1 ;
    _sumw2[i] = 0 ;
  }

  // Save real dimensions of dataset separately
  RooAbsArg* real ;
  _iterator->Reset() ;
  while((real=(RooAbsArg*)_iterator->Next())) {
    if (dynamic_cast<RooAbsReal*>(real)) _realVars.add(*real) ;
  }
  _realIter = _realVars.createIterator() ;

  if (!fillTree) return ;

  // Fill TTree with bin center coordinates
  // Calculate plot bins of components from master index
  Int_t ibin ;
  for (ibin=0 ; ibin<_arrSize ; ibin++) {
    _iterator->Reset() ;
    RooAbsLValue* arg ;
    Int_t i(0), idx(0), tmp(ibin) ;
    Double_t binVolume(1) ;
    while((arg=dynamic_cast<RooAbsLValue*>(_iterator->Next()))) {
      idx  = tmp / _idxMult[i] ;
      tmp -= idx*_idxMult[i++] ;
      RooAbsLValue* arglv = dynamic_cast<RooAbsLValue*>(arg) ;
      arglv->setBin(idx) ;
      binVolume *= arglv->getBinWidth(idx) ;
    }
    _binv[ibin] = binVolume ;
    Fill() ;
  }

}



RooDataHist::RooDataHist(const RooDataHist& other, const char* newname) :
  RooTreeData(other,newname), RooDirItem(), _curWeight(0), _curVolume(1), _pbinv(0)
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
  _errLo = new Double_t[_arrSize] ;
  _errHi = new Double_t[_arrSize] ;
  _binv = new Double_t[_arrSize] ;
  _sumw2 = new Double_t[_arrSize] ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = other._wgt[i] ;
    _errLo[i] = other._errLo[i] ;
    _errHi[i] = other._errHi[i] ;
    _sumw2[i] = other._sumw2[i] ;
    _binv[i] = other._binv[i] ;
  }  

  // Save real dimensions of dataset separately
  RooAbsArg* arg ;
  _iterator->Reset() ;
  while((arg=(RooAbsArg*)_iterator->Next())) {
    if (dynamic_cast<RooAbsReal*>(arg)) _realVars.add(*arg) ;
  }
  _realIter = _realVars.createIterator() ;

  appendToDir(this,kTRUE) ;
}


RooDataHist::RooDataHist(const char* name, const char* title, RooDataHist* h, const RooArgSet& varSubset, 
			 const RooFormulaVar* cutVar, const char* cutRange, Int_t nStart, Int_t nStop, Bool_t copyCache) :
  RooTreeData(name,title,h,varSubset,cutVar,cutRange,nStart,nStop,copyCache), _curWeight(0), _curVolume(1), _pbinv(0)
{
  // Constructor of a data hist from (part of) an existing data hist. The dimensions
  // of the data set are defined by the 'vars' RooArgSet, which can be identical
  // to 'dset' dimensions, or a subset thereof. Reduced dimensions will be projected
  // in the output data hist. The optional 'cutVar' formula variable can used to 
  // select the subset of bins to be copied.
  //
  // For most uses the RooAbsData::reduce() wrapper function, which uses this constructor, 
  // is the most convenient way to create a subset of an existing data

  initialize(kFALSE) ;

  // Copy weight array etc
  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = h->_wgt[i] ;
    _errLo[i] = h->_errLo[i] ;
    _errHi[i] = h->_errHi[i] ;
    _sumw2[i] = h->_sumw2[i] ;
    _binv[i] = h->_binv[i] ;
  }  

  appendToDir(this,kTRUE) ;
}

RooAbsData* RooDataHist::cacheClone(const RooArgSet* newCacheVars, const char* newName) 
{
  RooDataHist* dhist = new RooDataHist(newName?newName:GetName(),GetTitle(),this,*get(),0,0,0,2000000000,kTRUE) ; 

  RooArgSet* selCacheVars = (RooArgSet*) newCacheVars->selectCommon(dhist->_cachedVars) ;
  dhist->initCache(*selCacheVars) ;
  delete selCacheVars ;

  return dhist ;
}


RooAbsData* RooDataHist::reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange, 
				   Int_t nStart, Int_t nStop, Bool_t /*copyCache*/)
{
  // Implementation of RooAbsData virtual method that drives the RooAbsData::reduce() methods
  checkInit() ;

  RooArgSet* myVarSubset = (RooArgSet*) _vars.selectCommon(varSubset) ;
  RooDataHist *rdh = new RooDataHist(GetName(), GetTitle(), *myVarSubset) ;

  RooFormulaVar* cloneVar = 0;
  RooArgSet* tmp(0) ;
  if (cutVar) {
    // Deep clone cutVar and attach clone to this dataset
    tmp = (RooArgSet*) RooArgSet(*cutVar).snapshot() ;
    if (!tmp) {
      cout << "RooDataHist::reduceEng(" << GetName() << ") Couldn't deep-clone cut variable, abort," << endl ;
      return 0 ;
    }
    cloneVar = (RooFormulaVar*) tmp->find(cutVar->GetName()) ;
    cloneVar->attachDataSet(*this) ;
  }

  Int_t i ;
  Double_t lo,hi ;
  Int_t nevt = nStop < numEntries() ? nStop : numEntries() ;
  TIterator* vIter = get()->createIterator() ;
  for (i=nStart ; i<nevt ; i++) {
    const RooArgSet* row = get(i) ;

    Bool_t doSelect(kTRUE) ;
    if (cutRange) {
      RooAbsArg* arg ;
      vIter->Reset() ;
      while((arg=(RooAbsArg*)vIter->Next())) {	
	if (!arg->inRange(cutRange)) {
	  doSelect = kFALSE ;
	  break ;
	}
      }
    }
    if (!doSelect) continue ;

    if (!cloneVar || cloneVar->getVal()) {
      weightError(lo,hi,SumW2) ;
      rdh->add(*row,weight(),lo*lo) ;
    }
  }
  delete vIter ;

  if (cloneVar) {
    delete tmp ;
  } 
  
    return rdh ;
  }



RooDataHist::~RooDataHist() 
{
  // Destructor

  if (_wgt) delete[] _wgt ;
  if (_errLo) delete[] _errLo ;
  if (_errHi) delete[] _errHi ;
  if (_sumw2) delete[] _sumw2 ;
  if (_binv) delete[] _binv ;
  if (_pbinv) delete[] _pbinv ;
  if (_idxMult) delete[] _idxMult ;
  if (_realIter) delete _realIter ;

   removeFromDir(this) ;
}


Int_t RooDataHist::calcTreeIndex() const 
{
  // Calculate the index for the weights array corresponding to 
  // to the bin enclosing the current coordinates of the internal argset

  _iterator->Reset() ;
  RooAbsLValue* arg ;
  Int_t masterIdx(0), i(0) ;
  while((arg=dynamic_cast<RooAbsLValue*>(_iterator->Next()))) {
    masterIdx += _idxMult[i++]*arg->getBin() ;
  }
  return masterIdx ;
}


void RooDataHist::dump2() 
{
  // Debug stuff, should go...
  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    if (_wgt[i]!=0) {
      cout << "wgt[" << i << "] = " << _wgt[i] << "err[" << i << "] = " << _errLo[i] << " vol[" << i << "] = " << _binv[i] << endl ;
    }
  }
}



RooPlot *RooDataHist::plotOn(RooPlot *frame, PlotOpt o) const 
{
  if (o.bins) return RooTreeData::plotOn(frame,o) ;

  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  RooRealVar* dataVar = (RooRealVar*) _vars.find(var->GetName()) ;
  if (!dataVar) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: dataset doesn't contain plot frame variable" << endl;
    return 0;
  }

  o.bins = &dataVar->getBinning() ;
  return RooTreeData::plotOn(frame,o) ;
}




Double_t RooDataHist::weight(const RooArgSet& bin, Int_t intOrder, Bool_t correctForBinSize) 
{
  // Return the weight at given coordinates with optional
  // interpolation. If intOrder is zero, the weight
  // for the bin enclosing the coordinates
  // contained in 'bin' is returned. For higher values,
  // the result is interpolated in the real dimensions 
  // of the dataset
  // 

  // Handle illegal intOrder values
  if (intOrder<0) {
    cout << "RooDataHist::weight(" << GetName() << ") ERROR: interpolation order must be positive" << endl ;
    return 0 ;
  }

  // Handle no-interpolation case
  if (intOrder==0) {
    _vars = bin ;
    Int_t idx = calcTreeIndex() ;
    if (correctForBinSize) {
      return _wgt[idx] / _binv[idx] ;
    } else {
      return _wgt[idx] ;
    }
  }

  // Handle all interpolation cases
  _vars = bin ;

  Double_t wInt(0) ;
  if (_realVars.getSize()==1) {

    // 1-dimensional interpolation
    _realIter->Reset() ;
    RooRealVar* real=(RooRealVar*)_realIter->Next() ;
    wInt = interpolateDim(*real,((RooAbsReal*)bin.find(real->GetName()))->getVal(), intOrder, correctForBinSize) ;

  } else if (_realVars.getSize()==2) {

    // 2-dimensional interpolation
    _realIter->Reset() ;
    RooRealVar* realX=(RooRealVar*)_realIter->Next() ;
    RooRealVar* realY=(RooRealVar*)_realIter->Next() ;
    Double_t xval = ((RooAbsReal*)bin.find(realX->GetName()))->getVal() ;
    Double_t yval = ((RooAbsReal*)bin.find(realY->GetName()))->getVal() ;
    
    Int_t ybinC = realY->getBin() ;
    Int_t ybinLo = ybinC-intOrder/2 - ((yval<realY->getBinning().binCenter(ybinC))?1:0) ;
    Int_t ybinM = realY->numBins() ;
    
    Int_t i ;
    Double_t yarr[10] ;
    Double_t xarr[10] ;
    for (i=ybinLo ; i<=intOrder+ybinLo ; i++) {
      Int_t ibin ;
      if (i>=0 && i<ybinM) {
	// In range
	ibin = i ;
	realY->setBin(ibin) ;
	xarr[i-ybinLo] = realY->getVal() ;
      } else if (i>=ybinM) {
	// Overflow: mirror
	ibin = 2*ybinM-i-1 ;
	realY->setBin(ibin) ;
	xarr[i-ybinLo] = 2*realY->getMax()-realY->getVal() ;
      } else {
	// Underflow: mirror
	ibin = -i ;
	realY->setBin(ibin) ;
	xarr[i-ybinLo] = 2*realY->getMin()-realY->getVal() ;
      }
      yarr[i-ybinLo] = interpolateDim(*realX,xval,intOrder,correctForBinSize) ;	
    }
    wInt = RooMath::interpolate(xarr,yarr,intOrder+1,yval) ;
    
  } else {

    // Higher dimensional scenarios not yet implemented
    cout << "RooDataHist::weight(" << GetName() << ") interpolation in " 
	 << _realVars.getSize() << " dimensions not yet implemented" << endl ;
    return weight(bin,0) ;

  }

  // Cut off negative values
  if (wInt<0) wInt=0. ;

  return wInt ;
}




void RooDataHist::weightError(Double_t& lo, Double_t& hi, ErrorType etype) const 
{ 
  // Return error on current weight
  switch (etype) {
  case Poisson:
    if (_curWgtErrLo>=0) {
      // Weight is preset or precalculated    
      lo = _curWgtErrLo ;
      hi = _curWgtErrHi ;
      return ;
    }
    
    // Calculate poisson errors
    Double_t ym,yp ;  
    RooHistError::instance().getPoissonInterval(Int_t(weight()+0.5),ym,yp,1) ;
    _curWgtErrLo = weight()-ym ;
    _curWgtErrHi = yp-weight() ;
    _errLo[_curIndex] = _curWgtErrLo ;
    _errHi[_curIndex] = _curWgtErrHi ;
    lo = _curWgtErrLo ;
    hi = _curWgtErrHi ;
    return ;

  case SumW2:
    lo = sqrt(_curSumW2) ;
    hi = sqrt(_curSumW2) ;
    return ;
  }
}


// wve adjust for variable bin sizes
Double_t RooDataHist::interpolateDim(RooRealVar& dim, Double_t xval, Int_t intOrder, Bool_t correctForBinSize) 
{
  // Perform boundary safe 'intOrder'-th interpolation of weights in dimension 'dim'
  // at current value 'xval'

  // Fill workspace arrays spanning interpolation area
  Int_t fbinC = dim.getBin() ;
  Int_t fbinLo = fbinC-intOrder/2 - ((xval<dim.getBinning().binCenter(fbinC))?1:0) ;
  Int_t fbinM = dim.numBins() ;

  Int_t i ;
  Double_t yarr[10] ;
  Double_t xarr[10] ;
  for (i=fbinLo ; i<=intOrder+fbinLo ; i++) {
    Int_t ibin ;
    if (i>=0 && i<fbinM) {
      // In range
      ibin = i ;
      dim.setBin(ibin) ;
      xarr[i-fbinLo] = dim.getVal() ;
    } else if (i>=fbinM) {
      // Overflow: mirror
      ibin = 2*fbinM-i-1 ;
      dim.setBin(ibin) ;
      xarr[i-fbinLo] = 2*dim.getMax()-dim.getVal() ;
    } else {
      // Underflow: mirror
      ibin = -i ;
      dim.setBin(ibin) ;
      xarr[i-fbinLo] = 2*dim.getMin()-dim.getVal() ;
    }
    Int_t idx = calcTreeIndex() ;      
    yarr[i-fbinLo] = _wgt[idx] ; 
    if (correctForBinSize) yarr[i-fbinLo] /=  _binv[idx] ;
  }
  dim.setBin(fbinC) ;
  Double_t ret = RooMath::interpolate(xarr,yarr,intOrder+1,xval) ;
  return ret ;
}




void RooDataHist::add(const RooArgSet& row, Double_t weight, Double_t sumw2) 
{
  // Increment the weight of the bin enclosing the coordinates
  // given by 'row' by the specified amount
  checkInit() ;

  _vars = row ;
  Int_t idx = calcTreeIndex() ;
  _wgt[idx] += weight ;  
  _sumw2[idx] += (sumw2>0?sumw2:weight*weight) ;
  _errLo[idx] = -1 ;
  _errHi[idx] = -1 ;
}


void RooDataHist::set(const RooArgSet& row, Double_t weight, Double_t wgtErrLo, Double_t wgtErrHi) 
{
  // Increment the weight of the bin enclosing the coordinates
  // given by 'row' by the specified amount
  checkInit() ;

  _vars = row ;
  Int_t idx = calcTreeIndex() ;
  _wgt[idx] = weight ;  
  _errLo[idx] = wgtErrLo ;  
  _errHi[idx] = wgtErrHi ;  
}


void RooDataHist::set(const RooArgSet& row, Double_t weight, Double_t wgtErr) 
{
  // Increment the weight of the bin enclosing the coordinates
  // given by 'row' by the specified amount
  checkInit() ;

  _vars = row ;
  Int_t idx = calcTreeIndex() ;
  _wgt[idx] = weight ;  
  _errLo[idx] = wgtErr ;  
  _errHi[idx] = wgtErr ;  
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
  checkInit() ;

  RooFormulaVar* cloneVar = 0;
  RooArgSet* tmp(0) ;
  if (cutVar) {
    // Deep clone cutVar and attach clone to this dataset
    tmp = (RooArgSet*) RooArgSet(*cutVar).snapshot() ;
    if (!tmp) {
      cout << "RooDataHist::add(" << GetName() << ") Couldn't deep-clone cut variable, abort," << endl ;
      return ;
    }

    cloneVar = (RooFormulaVar*) tmp->find(cutVar->GetName()) ;
    cloneVar->attachDataSet(dset) ;
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

  Int_t i ;
  Double_t total(0) ;
  for (i=0 ; i<_arrSize ; i++) {
    
    Double_t binVolume = correctForBinSize ? _binv[i] : 1.0 ;
    total += _wgt[i]/binVolume ;
  }

  return total ;
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

  RooArgSet sliceOnlySet(sliceSet) ;
  sliceOnlySet.remove(sumSet,kTRUE,kTRUE) ;

  _vars = sliceOnlySet ;
  calculatePartialBinVolume(sliceOnlySet) ;

  TIterator* ssIter = sumSet.createIterator() ;
  
  // Calculate mask and refence plot bins for non-iterating variables
  RooAbsArg* arg ;
  Bool_t* mask = new Bool_t[_vars.getSize()] ;
  Int_t*  refBin = new Int_t[_vars.getSize()] ;

  Int_t i(0) ;
  _iterator->Reset() ;
  while((arg=(RooAbsArg*)_iterator->Next())) {
    if (sumSet.find(arg->GetName())) {
      mask[i] = kFALSE ;
    } else {
      mask[i] = kTRUE ;
      refBin[i] = (dynamic_cast<RooAbsLValue*>(arg))->getBin() ;
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
    while((!skip && (arg=(RooAbsArg*)_iterator->Next()))) {
      idx  = tmp / _idxMult[ivar] ;
      tmp -= idx*_idxMult[ivar] ;
      if (mask[ivar] && idx!=refBin[ivar]) skip=kTRUE ;
      ivar++ ;
    }
    
    if (!skip) {

      Double_t binVolume = correctForBinSize ? _pbinv[ibin] : 1.0 ;
      //cout << "ptotal += " << _wgt[ibin] << "/" << binVolume << endl ;
      total += _wgt[ibin]/binVolume ;
    }
  }
  delete ssIter ;

  delete[] mask ;
  delete[] refBin ;

  return total ;
}


void RooDataHist::calculatePartialBinVolume(const RooArgSet& dimSet) const 
{
  // Allocate cache if not yet existing
  if (_pbinv==0) {
    _pbinv = new Double_t[_arrSize] ;
  } else {

    // Check if partial bin volume for requested slice is in cache
    if (RooNameSet(dimSet)==_pbinvCache) {
      return ;
    }
  }

  // Calculate plot bins of components from master index
  Bool_t* selDim = new Bool_t[_vars.getSize()] ;
  _iterator->Reset() ;
  RooAbsArg* v ;
  Int_t i(0) ;
  while((v=(RooAbsArg*)_iterator->Next())) {
    selDim[i++] = dimSet.find(v->GetName()) ? kTRUE : kFALSE ;
  }

  // Recalculate partial bin volume cache
  Int_t ibin ;
  for (ibin=0 ; ibin<_arrSize ; ibin++) {
    _iterator->Reset() ;
    RooAbsLValue* arg ;
    Int_t i(0), idx(0), tmp(ibin) ;
    Double_t binVolume(1) ;
    while((arg=dynamic_cast<RooAbsLValue*>(_iterator->Next()))) {
      idx  = tmp / _idxMult[i] ;
      tmp -= idx*_idxMult[i++] ;
      if (selDim[i-1]) {
	RooAbsLValue* arglv = dynamic_cast<RooAbsLValue*>(arg) ;
	binVolume *= arglv->getBinWidth(idx) ;
      }
    }
    _pbinv[ibin] = binVolume ;
  }

  delete[] selDim ;
  
  // Update cache label
  _pbinvCache.refill(dimSet) ;
}



Int_t RooDataHist::numEntries(Bool_t useWeights) const 
{
  // Return the number of bins (useWeights=false) or
  // the sum of the weights of all bins (useWeight=true)

  if (!useWeights) return RooTreeData::numEntries() ;
  return Int_t(sumEntries()) ;
}


Double_t RooDataHist::sumEntries(const char* cutSpec, const char* cutRange) const
{
  if (cutSpec==0 && cutRange==0) {
    Int_t i ;
    Double_t n(0) ;
    for (i=0 ; i<_arrSize ; i++) {
      n+= _wgt[i] ;
    }
    return n ;
  } else {

    // Setup RooFormulaVar for cutSpec if it is present
    RooFormula* select = 0 ;
    if (cutSpec) {
      select = new RooFormula("select",cutSpec,*get()) ;
    }
    
    // Otherwise sum the weights in the event
    Double_t sumw(0) ;
    Int_t i ;
    for (i=0 ; i<GetEntries() ; i++) {
      get(i) ;
      if (select && select->eval()==0.) continue ;
      if (cutRange && !_vars.allInRange(cutRange)) continue ;
      sumw += weight() ;
    }
    
    if (select) delete select ;
    
    return sumw ;
  }
}



void RooDataHist::reset() 
{
  // Reset all bin weights to zero

  RooTreeData::reset() ;

  Int_t i ;
  for (i=0 ; i<_arrSize ; i++) {
    _wgt[i] = 0. ;
    _errLo[i] = -1 ;
    _errHi[i] = -1 ;
  }
  _curWeight = 0 ;
  _curWgtErrLo = -1 ;
  _curWgtErrHi = -1 ;
  _curVolume = 1 ;

}


const RooArgSet* RooDataHist::get(Int_t masterIdx) const  
{
  // Return an argset with the bin center coordinates for 
  // bin sequential number 'masterIdx'. For iterative use.

  _curWeight = _wgt[masterIdx] ;
  _curWgtErrLo = _errLo[masterIdx] ;
  _curWgtErrHi = _errHi[masterIdx] ;
  _curSumW2 = _sumw2[masterIdx] ;
  _curVolume = _binv[masterIdx] ; 
  _curIndex  = masterIdx ;
  return RooTreeData::get(masterIdx) ;  
}


const RooArgSet* RooDataHist::get(const RooArgSet& coord) const
{
  ((RooDataHist*)this)->_vars = coord ;
  return get(calcTreeIndex()) ;
}



Double_t RooDataHist::binVolume(const RooArgSet& coord) 
{
  ((RooDataHist*)this)->_vars = coord ;
  return _binv[calcTreeIndex()] ;
}






