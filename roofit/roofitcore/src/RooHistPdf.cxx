/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofit:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooHistPdf implements a probablity density function sampled from a 
// multidimensional histogram. The histogram distribution is explicitly
// normalized by RooHistPdf and can have an arbitrary number of real or 
// discrete dimensions.
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooHistPdf.h"
#include "RooDataHist.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooWorkspace.h"
#include "RooGlobalFunc.h"



using namespace std;

ClassImp(RooHistPdf)
;



//_____________________________________________________________________________
RooHistPdf::RooHistPdf() : _dataHist(0), _totVolume(0), _unitNorm(kFALSE)
{
  // Default constructor
  // coverity[UNINIT_CTOR]
  _histObsIter = _histObsList.createIterator() ;
  _pdfObsIter = _pdfObsList.createIterator() ;
}


//_____________________________________________________________________________
RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgSet& vars, 
		       const RooDataHist& dhist, Int_t intOrder) :
  RooAbsPdf(name,title), 
  _pdfObsList("pdfObs","List of p.d.f. observables",this),
  _dataHist((RooDataHist*)&dhist), 
  _codeReg(10),
  _intOrder(intOrder),
  _cdfBoundaries(kFALSE),
  _totVolume(0),
  _unitNorm(kFALSE)
{
  // Constructor from a RooDataHist. RooDataHist dimensions
  // can be either real or discrete. See RooDataHist::RooDataHist for details on the binning.
  // RooHistPdf neither owns or clone 'dhist' and the user must ensure the input histogram exists
  // for the entire life span of this PDF.

  _histObsList.addClone(vars) ;
  _pdfObsList.add(vars) ;

  // Verify that vars and dhist.get() have identical contents
  const RooArgSet* dvars = dhist.get() ;
  if (vars.getSize()!=dvars->getSize()) {
    coutE(InputArguments) << "RooHistPdf::ctor(" << GetName() 
			  << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
    assert(0) ;
  }
  TIterator* iter = vars.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dvars->find(arg->GetName())) {
      coutE(InputArguments) << "RooHistPdf::ctor(" << GetName() 
			    << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
      assert(0) ;
    }
  }
  delete iter ;

  _histObsIter = _histObsList.createIterator() ;
  _pdfObsIter = _pdfObsList.createIterator() ;


  // Adjust ranges of _histObsList to those of _dataHist 
  RooFIter oiter = _histObsList.fwdIterator() ;
  RooAbsArg* hobs ;
  while ((hobs = oiter.next())) {
    // Guaranteed to succeed, since checked above in ctor
    RooAbsArg* dhobs = dhist.get()->find(hobs->GetName()) ;
    RooRealVar* dhreal = dynamic_cast<RooRealVar*>(dhobs) ;
    if (dhreal){
      ((RooRealVar*)hobs)->setRange(dhreal->getMin(),dhreal->getMax()) ;
    }
  }
  
}




//_____________________________________________________________________________
RooHistPdf::RooHistPdf(const char *name, const char *title, const RooArgList& pdfObs, 
		       const RooArgList& histObs, const RooDataHist& dhist, Int_t intOrder) :
  RooAbsPdf(name,title), 
  _pdfObsList("pdfObs","List of p.d.f. observables",this),
  _dataHist((RooDataHist*)&dhist), 
  _codeReg(10),
  _intOrder(intOrder),
  _cdfBoundaries(kFALSE),
  _totVolume(0),
  _unitNorm(kFALSE)
{
  // Constructor from a RooDataHist. The first list of observables are the p.d.f.
  // observables, which may any RooAbsReal (function or variable). The second list
  // are the corresponding observables in the RooDataHist which must be of type
  // RooRealVar or RooCategory This constructor thus allows to apply a coordinate transformation
  // on the histogram data to be applied.

  _histObsList.addClone(histObs) ;
  _pdfObsList.add(pdfObs) ;

  // Verify that vars and dhist.get() have identical contents
  const RooArgSet* dvars = dhist.get() ;
  if (histObs.getSize()!=dvars->getSize()) {
    coutE(InputArguments) << "RooHistPdf::ctor(" << GetName() 
			  << ") ERROR histogram variable list and RooDataHist must contain the same variables." << endl ;
    throw(string("RooHistPdf::ctor() ERROR: histogram variable list and RooDataHist must contain the same variables")) ;
  }
  TIterator* iter = histObs.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dvars->find(arg->GetName())) {
      coutE(InputArguments) << "RooHistPdf::ctor(" << GetName() 
			    << ") ERROR variable list and RooDataHist must contain the same variables." << endl ;
      throw(string("RooHistPdf::ctor() ERROR: histogram variable list and RooDataHist must contain the same variables")) ;
    }
    if (!arg->isFundamental()) {
      coutE(InputArguments) << "RooHistPdf::ctor(" << GetName() 
			    << ") ERROR all elements of histogram observables set must be of type RooRealVar or RooCategory." << endl ;
      throw(string("RooHistPdf::ctor() ERROR all elements of histogram observables set must be of type RooRealVar or RooCategory.")) ;
    }
  }
  delete iter ;

  _histObsIter = _histObsList.createIterator() ;
  _pdfObsIter = _pdfObsList.createIterator() ;

  // Adjust ranges of _histObsList to those of _dataHist 
  RooFIter oiter = _histObsList.fwdIterator() ;
  RooAbsArg* hobs ;
  while ((hobs = oiter.next())) {
    // Guaranteed to succeed, since checked above in ctor
    RooAbsArg* dhobs = dhist.get()->find(hobs->GetName()) ;
    RooRealVar* dhreal = dynamic_cast<RooRealVar*>(dhobs) ;
    if (dhreal){
      ((RooRealVar*)hobs)->setRange(dhreal->getMin(),dhreal->getMax()) ;
    }
  }
}



//_____________________________________________________________________________
RooHistPdf::RooHistPdf(const RooHistPdf& other, const char* name) :
  RooAbsPdf(other,name), 
  _pdfObsList("pdfObs",this,other._pdfObsList),
  _dataHist(other._dataHist),
  _codeReg(other._codeReg),
  _intOrder(other._intOrder),
  _cdfBoundaries(other._cdfBoundaries),
  _totVolume(other._totVolume),
  _unitNorm(other._unitNorm)
{
  // Copy constructor

  _histObsList.addClone(other._histObsList) ;

  _histObsIter = _histObsList.createIterator() ;
  _pdfObsIter = _pdfObsList.createIterator() ;
}




//_____________________________________________________________________________
RooHistPdf::~RooHistPdf()
{
  // Destructor

  delete _histObsIter ;
  delete _pdfObsIter ;
}





//_____________________________________________________________________________
Double_t RooHistPdf::evaluate() const
{
  // Return the current value: The value of the bin enclosing the current coordinates
  // of the observables, normalized by the histograms contents. Interpolation
  // is applied if the RooHistPdf is configured to do that

  // Transfer values from   
  if (_pdfObsList.getSize()>0) {
    _histObsIter->Reset() ;
    _pdfObsIter->Reset() ;
    RooAbsArg* harg, *parg ;
    while((harg=(RooAbsArg*)_histObsIter->Next())) {
      parg = (RooAbsArg*)_pdfObsIter->Next() ;
      if (harg != parg) {
	parg->syncCache() ;
	harg->copyCache(parg,kTRUE) ;
	if (!harg->inRange(0)) {
	  return 0 ;
	}
      }
    }
  }

  Double_t ret =  _dataHist->weight(_histObsList,_intOrder,_unitNorm?kFALSE:kTRUE,_cdfBoundaries) ;  
  if (ret<0) {
    ret=0 ;
  }  
  return ret ;
}


//_____________________________________________________________________________
Double_t RooHistPdf::totVolume() const
{
  // Return the total volume spanned by the observables of the RooHistPdf

  // Return previously calculated value, if any
  if (_totVolume>0) {
    return _totVolume ;
  }
  _totVolume = 1. ;
  TIterator* iter = _histObsList.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooRealVar* real = dynamic_cast<RooRealVar*>(arg) ;
    if (real) {
      _totVolume *= (real->getMax()-real->getMin()) ;
    } else {
      RooCategory* cat = dynamic_cast<RooCategory*>(arg) ;
      if (cat) {
	_totVolume *= cat->numTypes() ;
      }
    }
  }
  delete iter ;
  return _totVolume ;
}

namespace {
    bool fullRange(const RooAbsArg& x ,const char* range)  {
      if (range == 0 || strlen(range) == 0 ) return true;
      const RooAbsRealLValue *_x = dynamic_cast<const RooAbsRealLValue*>(&x);
      if (!_x) return false;
      return ( _x->getMin(range) == _x->getMin() && _x->getMax(range) == _x->getMax() ) ; 
    }
}


//_____________________________________________________________________________
Int_t RooHistPdf::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const 
{
  // Determine integration scenario. If no interpolation is used,
  // RooHistPdf can perform all integrals over its dependents
  // analytically via partial or complete summation of the input
  // histogram. If interpolation is used on the integral over
  // all histogram observables is supported


  // First make list of pdf observables to histogram observables
  // and select only those for which the integral is over the full range
  RooArgList hobsl(_histObsList),pobsl(_pdfObsList) ;
  RooArgSet allVarsHist ;
  TIterator* iter = allVars.createIterator() ;
  RooAbsArg* pdfobs ;
  while((pdfobs=(RooAbsArg*)iter->Next())) {
    Int_t idx = pobsl.index(pdfobs) ;
    if (idx>=0) {
      RooAbsArg* hobs = hobsl.at(idx) ;
      if (hobs && fullRange( *hobs, rangeName ) ) {
	allVarsHist.add(*hobs) ;
      }
    }
  }
  delete iter ;

  // Simplest scenario, integrate over all dependents
  RooAbsCollection *allVarsCommon = allVarsHist.selectCommon(_histObsList) ;  
  Bool_t intAllObs = (allVarsCommon->getSize()==_histObsList.getSize()) ;
  delete allVarsCommon ;
  if (intAllObs) {
    analVars.add(allVars) ;
    return 1000 ;
  }

  // Disable partial analytical integrals if interpolation is used
//   if (_intOrder>0) {
//     return 0 ;
//   }

  // Find subset of _histObsList that integration is requested over
  RooArgSet* allVarsSel = (RooArgSet*) allVarsHist.selectCommon(_histObsList) ;
  if (allVarsSel->getSize()==0) {
    delete allVarsSel ;
    return 0 ;
  }

  // Partial integration scenarios.
  // Build unique code from bit mask of integrated variables in depList
  Int_t code(0),n(0) ;
  iter = _histObsList.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (allVars.find(arg->GetName())) {
      code |= (1<<n) ;
      analVars.add(*pobsl.at(n)) ;
    }
    n++ ;
  }
  delete iter ;

  return code ;

}



//_____________________________________________________________________________
Double_t RooHistPdf::analyticalIntegral(Int_t code, const char* /*rangeName*/) const 
{
  // Return integral identified by 'code'. The actual integration
  // is deferred to RooDataHist::sum() which implements partial
  // or complete summation over the histograms contents

  // WVE needs adaptation for rangeName feature
  // Simplest scenario, integration over all dependents
  if (code==1000) {    
    return _dataHist->sum(kFALSE) ;
  }

  // Partial integration scenario, retrieve set of variables, calculate partial sum
  RooArgSet intSet ;
  TIterator* iter = _histObsList.createIterator() ;
  RooAbsArg* arg ;
  Int_t n(0) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (code & (1<<n)) {
      intSet.add(*arg) ;
    }
    n++ ;
  }
  delete iter ;

  // WVE must sync hist slice list values to pdf slice list
  // Transfer values from   
  if (_pdfObsList.getSize()>0) {
    _histObsIter->Reset() ;
    _pdfObsIter->Reset() ;
    RooAbsArg* harg, *parg ;
    while((harg=(RooAbsArg*)_histObsIter->Next())) {
      parg = (RooAbsArg*)_pdfObsIter->Next() ;
      if (harg != parg) {
	parg->syncCache() ;
	harg->copyCache(parg,kTRUE) ;
      }
    }
  }  


  Double_t ret =  _dataHist->sum(intSet,_histObsList,kTRUE,kTRUE) ;

//    cout << "intSet = " << intSet << endl ;
//    cout << "slice position = " << endl ;
//    _histObsList.Print("v") ;
//    cout << "RooHistPdf::ai(" << GetName() << ") code = " << code << " ret = " << ret << endl ;

  return ret ;
}



//_____________________________________________________________________________
list<Double_t>* RooHistPdf::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const
{
  // Return sampling hint for making curves of (projections) of this function
  // as the recursive division strategy of RooCurve cannot deal efficiently
  // with the vertical lines that occur in a non-interpolated histogram

  // No hints are required when interpolation is used
  if (_intOrder>0) {
    return 0 ;
  }

  // Check that observable is in dataset, if not no hint is generated
  _histObsIter->Reset() ;
  _pdfObsIter->Reset() ;
  RooAbsArg *pdfObs, *histObs, *dhObs(0) ;
  while ((pdfObs = (RooAbsArg*)_pdfObsIter->Next()) && !dhObs) {
    histObs = (RooAbsArg*) _histObsIter->Next() ;
    if (TString(obs.GetName())==pdfObs->GetName()) {
      dhObs = _dataHist->get()->find(histObs->GetName()) ;
    }
  }

  if (!dhObs) {
    return 0 ;
  }
  RooAbsLValue* lval = dynamic_cast<RooAbsLValue*>(dhObs) ;
  if (!lval) {
    return 0 ;
  }

  // Retrieve position of all bin boundaries
  
  const RooAbsBinning* binning = lval->getBinningPtr(0) ;
  Double_t* boundaries = binning->array() ;

  list<Double_t>* hint = new list<Double_t> ;

  // Widen range slighty
  xlo = xlo - 0.01*(xhi-xlo) ;
  xhi = xhi + 0.01*(xhi-xlo) ;

  Double_t delta = (xhi-xlo)*1e-8 ;
 
  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]-delta) ;
      hint->push_back(boundaries[i]+delta) ;
    }
  }

  return hint ;
}



//______________________________________________________________________________
std::list<Double_t>* RooHistPdf::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const 
{
  // Return sampling hint for making curves of (projections) of this function
  // as the recursive division strategy of RooCurve cannot deal efficiently
  // with the vertical lines that occur in a non-interpolated histogram

  // No hints are required when interpolation is used
  if (_intOrder>0) {
    return 0 ;
  }

  // Check that observable is in dataset, if not no hint is generated
  RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(_dataHist->get()->find(obs.GetName())) ;
  if (!lvarg) {
    return 0 ;
  }

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(0) ;
  Double_t* boundaries = binning->array() ;

  list<Double_t>* hint = new list<Double_t> ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]) ;
    }
  }

  return hint ;
}




//_____________________________________________________________________________
Int_t RooHistPdf::getMaxVal(const RooArgSet& vars) const 
{
  // Only handle case of maximum in all variables
  RooAbsCollection* common = _pdfObsList.selectCommon(vars) ;
  if (common->getSize()==_pdfObsList.getSize()) {
    delete common ;
    return 1;
  }
  delete common ;
  return 0 ;
}


//_____________________________________________________________________________
Double_t RooHistPdf::maxVal(Int_t code) const 
{
  assert(code==1) ;

  Double_t max(-1) ;
  for (Int_t i=0 ; i<_dataHist->numEntries() ; i++) {
    _dataHist->get(i) ;
    Double_t wgt = _dataHist->weight() ;
    if (wgt>max) max=wgt ;
  }

  return max*1.05 ;
}




//_____________________________________________________________________________
Bool_t RooHistPdf::areIdentical(const RooDataHist& dh1, const RooDataHist& dh2) 
{
  if (fabs(dh1.sumEntries()-dh2.sumEntries())>1e-8) return kFALSE ;
  if (dh1.numEntries() != dh2.numEntries()) return kFALSE ;
  for (int i=0 ; i < dh1.numEntries() ; i++) {
    dh1.get(i) ;
    dh2.get(i) ;
    if (fabs(dh1.weight()-dh2.weight())>1e-8) return kFALSE ;
  }
  return kTRUE ;
}



//_____________________________________________________________________________
Bool_t RooHistPdf::importWorkspaceHook(RooWorkspace& ws) 
{  
  // Check if our datahist is already in the workspace
  std::list<RooAbsData*> allData = ws.allData() ;
  std::list<RooAbsData*>::const_iterator iter ;
  for (iter = allData.begin() ; iter != allData.end() ; ++iter) {
    // If your dataset is already in this workspace nothing needs to be done
    if (*iter == _dataHist) {
      return kFALSE ;
    }
  }

  // Check if dataset with given name already exists
  RooAbsData* wsdata = ws.data(_dataHist->GetName()) ;

  if (wsdata) {

    // Yes it exists - now check if it is identical to our internal histogram 
    if (wsdata->InheritsFrom(RooDataHist::Class())) {

      // Check if histograms are identical
      if (areIdentical((RooDataHist&)*wsdata,*_dataHist)) {

	// Exists and is of correct type, and identical -- adjust internal pointer to WS copy
	_dataHist = (RooDataHist*) wsdata ;
      } else {

	// not identical, clone rename and import
	TString uniqueName = Form("%s_%s",_dataHist->GetName(),GetName()) ;
	Bool_t flag = ws.import(*_dataHist,RooFit::Rename(uniqueName.Data())) ;
	if (flag) {
	  coutE(ObjectHandling) << " RooHistPdf::importWorkspaceHook(" << GetName() << ") unable to import clone of underlying RooDataHist with unique name " << uniqueName << ", abort" << endl ;
	  return kTRUE ;
	}
	_dataHist = (RooDataHist*) ws.data(uniqueName.Data()) ;
      }

    } else {

      // Exists and is NOT of correct type: clone rename and import
      TString uniqueName = Form("%s_%s",_dataHist->GetName(),GetName()) ;
      Bool_t flag = ws.import(*_dataHist,RooFit::Rename(uniqueName.Data())) ;
      if (flag) {
	coutE(ObjectHandling) << " RooHistPdf::importWorkspaceHook(" << GetName() << ") unable to import clone of underlying RooDataHist with unique name " << uniqueName << ", abort" << endl ;
	return kTRUE ;
      }
      _dataHist = (RooDataHist*) ws.data(uniqueName.Data()) ;
      
    }
    return kFALSE ;
  }
  
  // We need to import our datahist into the workspace
  ws.import(*_dataHist) ;

  // Redirect our internal pointer to the copy in the workspace
  _dataHist = (RooDataHist*) ws.data(_dataHist->GetName()) ;
  return kFALSE ;
}


//______________________________________________________________________________
void RooHistPdf::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooHistPdf.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RooHistPdf::Class(),this);
      // WVE - interim solution - fix proxies here
      //_proxyList.Clear() ;
      //registerProxy(_pdfObsList) ;
   } else {
      R__b.WriteClassBuffer(RooHistPdf::Class(),this);
   }
}

