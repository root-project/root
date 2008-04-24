/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

// -- CLASS DESCRIPTION [PDF] --
// RooAddPdf is an efficient implementation of a sum of PDFs of the form 
//
//  c_1*PDF_1 + c_2*PDF_2 + ... c_n*PDF_n 
//
// or 
//
//  c_1*PDF_1 + c_2*PDF_2 + ... (1-sum(c_1...c_n-1))*PDF_n 
//
// The first form is for extended likelihood fits, where the
// expected number of events is Sum(i) c_i. The coefficients c_i
// can either be explicitly provided, or, if all components support
// extended likelihood fits, they can be calculated the contribution
// of each PDF to the total number of expected events.
//
// In the second form, the sum of the coefficients is enforced to be one,
// and the coefficient of the last PDF is calculated from that condition.
//
// RooAddPdf relies on each component PDF to be normalized and will perform 
// no normalization other than calculating the proper last coefficient c_n, if requested.
// An (enforced) condition for this assuption is that each PDF_i is independent
// of each coefficient_i.
//
// 

#include "RooFit.h"
#include "RooMsgService.h"

#include "TIterator.h"
#include "TIterator.h"
#include "TList.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooRealProxy.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooAddGenContext.h"
#include "RooRealConstant.h"
#include "RooNameReg.h"
#include "RooMsgService.h"
#include "RooRecursiveFraction.h"

#include "Riostream.h"


ClassImp(RooAddPdf)
;

RooAddPdf::RooAddPdf() :
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _codeReg(10),
  _snormList(0)
{
  _pdfIter   = _pdfList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;

  _coefCache = new Double_t[10] ;
  _coefErrCount = _errorCount ;
}


RooAddPdf::RooAddPdf(const char *name, const char *title) :
  RooAbsPdf(name,title), 
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _codeReg(10),
  _pdfList("pdfs","List of PDFs",this),
  _coefList("coefficients","List of coefficients",this),
  _snormList(0),
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE)
{
  // Dummy constructor 
  _pdfIter   = _pdfList.createIterator() ;
  _coefIter  = _coefList.createIterator() ;

  _coefCache = new Double_t[10] ;
  _coefErrCount = _errorCount ;

}


RooAddPdf::RooAddPdf(const char *name, const char *title,
		     RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) : 
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _codeReg(10),
  _pdfList("pdfs","List of PDFs",this),
  _coefList("coefficients","List of coefficients",this),
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE)
{
  // Special constructor with two PDFs and one coefficient (most frequent use case)

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;

  _pdfList.add(pdf1) ;  
  _pdfList.add(pdf2) ;
  _coefList.add(coef1) ;

  _coefCache = new Double_t[_pdfList.getSize()] ;
  _coefErrCount = _errorCount ;
}


RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& inPdfList, const RooArgList& inCoefList, Bool_t recursiveFractions) :
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _codeReg(10),
  _pdfList("pdfs","List of PDFs",this),
  _coefList("coefficients","List of coefficients",this),
  _haveLastCoef(kFALSE),
  _allExtendable(kFALSE)
{ 
  // Generic constructor from list of PDFs and list of coefficients.
  // Each pdf list element (i) is paired with coefficient list element (i).
  // The number of coefficients must be either equal to the number of PDFs,
  // in which case extended MLL fitting is enabled, or be one less.
  //
  // All PDFs must inherit from RooAbsPdf. All coefficients must inherit from RooAbsReal

  if (inPdfList.getSize()>inCoefList.getSize()+1 || inPdfList.getSize()<inCoefList.getSize()) {
    coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() 
			  << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1" << endl ;
    assert(0) ;
  }

  if (recursiveFractions && inPdfList.getSize()!=inCoefList.getSize()+1) {
    coutW(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() 
			  << ") WARNING inconsistent input: recursive fractions options can only be used if Npdf=Ncoef+1, ignoring recursive fraction setting" << endl ;
  }


  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N PDFs and N or N-1 coefs
  TIterator* pdfIter = inPdfList.createIterator() ;
  TIterator* coefIter = inCoefList.createIterator() ;
  RooAbsPdf* pdf ;
  RooAbsReal* coef ;

  RooArgList partinCoefList ;

  while((coef = (RooAbsPdf*)coefIter->Next())) {
    pdf = (RooAbsPdf*) pdfIter->Next() ;
    if (!pdf) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() 
			    << ") number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1" << endl ;
      assert(0) ;
    }
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") coefficient " << coef->GetName() << " is not of type RooAbsReal, ignored" << endl ;
      continue ;
    }
    if (!dynamic_cast<RooAbsReal*>(pdf)) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not of type RooAbsPdf, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;

    if (recursiveFractions) {
      partinCoefList.add(*coef) ;
      RooAbsReal* rfrac = new RooRecursiveFraction(Form("%s_recursive_fraction_%s",GetName(),pdf->GetName()),"Recursive Fraction",partinCoefList) ;
      addOwnedComponents(*rfrac) ;
      _coefList.add(*rfrac) ;
    } else {
      _coefList.add(*coef) ;    
    }
  }

  pdf = (RooAbsPdf*) pdfIter->Next() ;
  if (pdf) {
    if (!dynamic_cast<RooAbsReal*>(pdf)) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") last pdf " << coef->GetName() << " is not of type RooAbsPdf, fatal error" << endl ;
      assert(0) ;
    }
    _pdfList.add(*pdf) ;  
  } else {
    _haveLastCoef=kTRUE ;
  }

  delete pdfIter ;
  delete coefIter  ;

  _coefCache = new Double_t[_pdfList.getSize()] ;
  _coefErrCount = _errorCount ;
}



RooAddPdf::RooAddPdf(const char *name, const char *title, const RooArgList& inPdfList) :
  RooAbsPdf(name,title),
  _refCoefNorm("!refCoefNorm","Reference coefficient normalization set",this,kFALSE,kFALSE),
  _refCoefRangeName(0),
  _projectCoefs(kFALSE),
  _projCacheMgr(this,10),
  _pdfList("pdfs","List of PDFs",this),
  _coefList("coefficients","List of coefficients",this),
  _haveLastCoef(kFALSE),
  _allExtendable(kTRUE)
{ 
  // Generic constructor from list of extended PDFs. There are no coefficients as the expected
  // number of events from each components determine the relative weight of the PDFs.
  // 
  // All PDFs must inherit from RooAbsPdf. 

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
 
  // Constructor with N PDFs 
  TIterator* pdfIter = inPdfList.createIterator() ;
  RooAbsPdf* pdf ;
  while((pdf = (RooAbsPdf*) pdfIter->Next())) {
    
    if (!dynamic_cast<RooAbsReal*>(pdf)) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not of type RooAbsPdf, ignored" << endl ;
      continue ;
    }
    if (!pdf->canBeExtended()) {
      coutE(InputArguments) << "RooAddPdf::RooAddPdf(" << GetName() << ") pdf " << pdf->GetName() << " is not extendable, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;    
  }

  delete pdfIter ;

  _coefCache = new Double_t[_pdfList.getSize()] ;
  _coefErrCount = _errorCount ;
}



RooAddPdf::RooAddPdf(const RooAddPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _refCoefNorm("!refCoefNorm",this,other._refCoefNorm),
  _refCoefRangeName((TNamed*)other._refCoefRangeName),
  _projectCoefs(other._projectCoefs),
  _projCacheMgr(other._projCacheMgr,this),
  _codeReg(other._codeReg),
  _pdfList("pdfs",this,other._pdfList),
  _coefList("coefficients",this,other._coefList),
  _haveLastCoef(other._haveLastCoef),
  _allExtendable(other._allExtendable)
{
  // Copy constructor

  _pdfIter  = _pdfList.createIterator() ;
  _coefIter = _coefList.createIterator() ;
  _coefCache = new Double_t[_pdfList.getSize()] ;
  _coefErrCount = _errorCount ;
}


RooAddPdf::~RooAddPdf()
{
  // Destructor
  delete _pdfIter ;
  delete _coefIter ;

  if (_coefCache) delete[] _coefCache ;
}



void RooAddPdf::fixCoefNormalization(const RooArgSet& refCoefNorm) 
{
  if (refCoefNorm.getSize()==0) {
    _projectCoefs = kFALSE ;
    return ;
  }
  _projectCoefs = kTRUE ;  

  _refCoefNorm.removeAll() ;
  _refCoefNorm.add(refCoefNorm) ;

  _projCacheMgr.reset() ;
}


void RooAddPdf::fixCoefRange(const char* rangeName)
{
  _refCoefRangeName = (TNamed*)RooNameReg::ptr(rangeName) ;
  if (_refCoefRangeName) _projectCoefs = kTRUE ;
}





RooAddPdf::CacheElem* RooAddPdf::getProjCache(const RooArgSet* nset, const RooArgSet* iset, const char* rangeName) const
{
  // Check if cache already exists 
  CacheElem* cache = (CacheElem*) _projCacheMgr.getObj(nset,iset,0,RooNameReg::ptr(rangeName)) ;
  if (cache) {
    return cache ;
  }

  //Create new cache 
  cache = new CacheElem ;

  // *** PART 1 : Create supplemental normalization list ***

  // Retrieve the combined set of dependents of this PDF ;
  RooArgSet *fullDepList = getObservables(nset) ;
  if (iset) {
    fullDepList->remove(*iset,kTRUE,kTRUE) ;
  }    

  // Fill with dummy unit RRVs for now
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsPdf* pdf ;
  RooAbsReal* coef ;
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {    
    coef=(RooAbsPdf*)_coefIter->Next() ;

    // Start with full list of dependents
    RooArgSet supNSet(*fullDepList) ;

    // Remove PDF dependents
    RooArgSet* pdfDeps = pdf->getObservables(nset) ;
    if (pdfDeps) {
      supNSet.remove(*pdfDeps,kTRUE,kTRUE) ;
      delete pdfDeps ; 
    }

    // Remove coef dependents
    RooArgSet* coefDeps = coef ? coef->getObservables(nset) : 0 ;
    if (coefDeps) {
      supNSet.remove(*coefDeps,kTRUE,kTRUE) ;
      delete coefDeps ;
    }
    
    RooAbsReal* snorm ;
    TString name(GetName()) ;
    name.Append("_") ;
    name.Append(pdf->GetName()) ;
    name.Append("_SupNorm") ;
    if (supNSet.getSize()>0) {
      snorm = new RooRealIntegral(name,"Supplemental normalization integral",RooRealConstant::value(1.0),supNSet) ;
    } else {
      snorm = new RooRealVar(name,"Unit Supplemental normalization integral",1.0) ;
    }
    cache->_suppNormList.addOwned(*snorm) ;
  }

  delete fullDepList ;
    
  if (_verboseEval>1) {
    cxcoutD(Caching) << "RooAddPdf::syncSuppNormList(" << GetName() << ") synching supplemental normalization list for norm" << (nset?*nset:RooArgSet()) << endl ;
    if dologD(Caching) {
      cache->_suppNormList.Print("v") ;
    }
  }


  // *** PART 2 : Create projection coefficients ***

  // If no projections required stop here
  if (!_projectCoefs) {
    _projCacheMgr.setObj(nset,iset,cache,RooNameReg::ptr(rangeName)) ;
    return cache ;
  }


  // Reduce iset/nset to actual dependents of this PDF
  RooArgSet* nset2 = nset ? getObservables(nset) : new RooArgSet() ;

  // Check if requested transformation is not identity 
  if (!nset2->equals(_refCoefNorm) || _refCoefRangeName !=0 || rangeName !=0 ) {
   
    coutI(Caching) << "RooAddPdf::syncCoefProjList(" << GetName() << ") creating coefficient projection integrals" << endl
		   << "  from current normalization: " << *nset2 << endl
		   << "          with current range: " << (rangeName?rangeName:"<none>") << endl 
		   << "  to reference normalization: "  << _refCoefNorm << endl
		   << "        with reference range: " << (_refCoefRangeName?RooNameReg::str(_refCoefRangeName):"<none>") << endl ;
    
    // Recalculate projection integrals of PDFs 
    _pdfIter->Reset() ;
    RooAbsPdf* thePdf ;

    while((thePdf=(RooAbsPdf*)_pdfIter->Next())) {

      // Calculate projection integral
      RooAbsReal* pdfProj ;
      if (!nset2->equals(_refCoefNorm)) {
	pdfProj = thePdf->createIntegral(*nset2,_refCoefNorm) ;
	pdfProj->setOperMode(operMode()) ;
      } else {
	TString name(GetName()) ;
	name.Append("_") ;
	name.Append(thePdf->GetName()) ;
	name.Append("_ProjectNorm") ;
	pdfProj = new RooRealVar(name,"Unit Projection normalization integral",1.0) ;
      }

      cache->_projList.addOwned(*pdfProj) ;

      // Calculation optional supplemental normalization term
      RooArgSet supNormSet(_refCoefNorm) ;
      RooArgSet* deps = thePdf->getParameters(RooArgSet()) ;
      supNormSet.remove(*deps,kTRUE,kTRUE) ;
      delete deps ;

      RooAbsReal* snorm ;
      TString name(GetName()) ;
      name.Append("_") ;
      name.Append(thePdf->GetName()) ;
      name.Append("_ProjSupNorm") ;
      if (supNormSet.getSize()>0) {
	snorm = new RooRealIntegral(name,"Projection Supplemental normalization integral",
				    RooRealConstant::value(1.0),supNormSet) ;
      } else {
	snorm = new RooRealVar(name,"Unit Projection Supplemental normalization integral",1.0) ;
      }
      cache->_suppProjList.addOwned(*snorm) ;

      // Calculate reference range adjusted projection integral
      RooAbsReal* rangeProj1 ;
      if (_refCoefRangeName && _refCoefNorm.getSize()>0) {
	rangeProj1 = thePdf->createIntegral(_refCoefNorm,_refCoefNorm,RooNameReg::str(_refCoefRangeName)) ;
	rangeProj1->setOperMode(operMode()) ;
      } else {
	TString theName(GetName()) ;
	theName.Append("_") ;
	theName.Append(thePdf->GetName()) ;
	theName.Append("_RangeNorm1") ;
	rangeProj1 = new RooRealVar(theName,"Unit range normalization integral",1.0) ;
      }
      cache->_refRangeProjList.addOwned(*rangeProj1) ;
      

      // Calculate range adjusted projection integral
      RooAbsReal* rangeProj2 ;
      if (rangeName && _refCoefNorm.getSize()>0) {
	rangeProj2 = thePdf->createIntegral(_refCoefNorm,_refCoefNorm,rangeName) ;
	rangeProj2->setOperMode(operMode()) ;
      } else {
	TString theName(GetName()) ;
	theName.Append("_") ;
	theName.Append(thePdf->GetName()) ;
	theName.Append("_RangeNorm2") ;
	rangeProj2 = new RooRealVar(theName,"Unit range normalization integral",1.0) ;
      }
      cache->_rangeProjList.addOwned(*rangeProj2) ;

    }               

  }

  delete nset2 ;

  _projCacheMgr.setObj(nset,iset,cache,RooNameReg::ptr(rangeName)) ;

  return cache ;
}


void RooAddPdf::updateCoefficients(CacheElem& cache, const RooArgSet* nset) const 
{

  // cxcoutD(ChangeTracking) << "RooAddPdf::updateCoefficients(" << GetName() << ") update coefficients" << endl ;
  
  Int_t i ;

  // Straight coefficients
  if (_allExtendable) {
    
    // coef[i] = expectedEvents[i] / SUM(expectedEvents)
    Double_t coefSum(0) ;
    for (i=0 ; i<_pdfList.getSize() ; i++) {
      _coefCache[i] = ((RooAbsPdf*)_pdfList.at(i))->expectedEvents(_refCoefNorm.getSize()>0?&_refCoefNorm:nset) ;
      coefSum += _coefCache[i] ;
    }
    if (coefSum==0.) {
      coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName() << ") WARNING: total number of expected events is 0" << endl ;
    } else {
      for (i=0 ; i<_pdfList.getSize() ; i++) {
	_coefCache[i] /= coefSum ;
      }			            
    }
    
  } else {
    if (_haveLastCoef) {
      
      // coef[i] = coef[i] / SUM(coef)
      Double_t coefSum(0) ;
      for (i=0 ; i<_coefList.getSize() ; i++) {
	_coefCache[i] = ((RooAbsPdf*)_coefList.at(i))->getVal(nset) ;
	coefSum += _coefCache[i] ;
      }		
      for (i=0 ; i<_coefList.getSize() ; i++) {
	_coefCache[i] /= coefSum ;
      }			
    } else {
      
      // coef[i] = coef[i] ; coef[n] = 1-SUM(coef[0...n-1])
      Double_t lastCoef(1) ;
      for (i=0 ; i<_coefList.getSize() ; i++) {
	_coefCache[i] = ((RooAbsPdf*)_coefList.at(i))->getVal(nset) ;
 	cxcoutD(Caching) << "SYNC: orig coef[" << i << "] = " << _coefCache[i] << endl ;
	lastCoef -= _coefCache[i] ;
      }			
      _coefCache[_coefList.getSize()] = lastCoef ;
      cxcoutD(Caching) << "SYNC: orig coef[" << _coefList.getSize() << "] = " << _coefCache[_coefList.getSize()] << endl ;
      
      
      // Warn about coefficient degeneration
      if ((lastCoef<-1e-05 || (lastCoef-1)>1e-5) && _coefErrCount-->0) {
	coutW(Eval) << "RooAddPdf::updateCoefCache(" << GetName() 
		    << " WARNING: sum of PDF coefficients not in range [0-1], value=" 
		    << 1-lastCoef ; 
	if (_coefErrCount==0) {
	  coutW(Eval) << " (no more will be printed)"  ;
	}
	coutW(Eval) << endl ;
      } 
    }
  }

  

  // Stop here if not projection is required or needed
  if ((!_projectCoefs) || cache._projList.getSize()==0) {
    //     cout << "SYNC no projection required rangeName = " << (rangeName?rangeName:"<none>") << endl ;
    return ;
  }

  // Adjust coefficients for given projection
  Double_t coefSum(0) ;
  for (i=0 ; i<_pdfList.getSize() ; i++) {
    RooAbsPdf::globalSelectComp(kTRUE) ;    

    RooAbsReal* pp = ((RooAbsReal*)cache._projList.at(i)) ; 
    RooAbsReal* sn = ((RooAbsReal*)cache._suppProjList.at(i)) ; 
    RooAbsReal* r1 = ((RooAbsReal*)cache._refRangeProjList.at(i)) ;
    RooAbsReal* r2 = ((RooAbsReal*)cache._rangeProjList.at(i)) ;
    
    cxcoutD(Caching) << "pp = " << pp->GetName() << endl 
		     << "sn = " << sn->GetName() << endl 
		     << "r1 = " << r1->GetName() << endl 
		     << "r2 = " << r2->GetName() << endl ;
    if (dologD(Caching)) {
      r1->printStream(ccoutD(Caching),0,kVerbose) ;
      r1->printCompactTree(ccoutD(Caching)) ;
    }

    Double_t proj = pp->getVal()/sn->getVal()*(r2->getVal()/r1->getVal()) ;  
    
    RooAbsPdf::globalSelectComp(kFALSE) ;

    _coefCache[i] *= proj ;
    coefSum += _coefCache[i] ;
  }
  for (i=0 ; i<_pdfList.getSize() ; i++) {
    _coefCache[i] /= coefSum ;
//     cout << "POST-SYNC coef[" << i << "] = " << _coefCache[i] << endl ;
  }
   

  
}



Double_t RooAddPdf::evaluate() const 
{
  // Calculate the current value
  const RooArgSet* nset = _normSet ; 
  CacheElem* cache = getProjCache(nset) ;

  updateCoefficients(*cache,nset) ;
  
  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsPdf* pdf ;

  Double_t snormVal ;
  Double_t value(0) ;
  Int_t i(0) ;
  while((pdf = (RooAbsPdf*)_pdfIter->Next())) {
    if (_coefCache[i]!=0.) {
      snormVal = nset ? ((RooAbsReal*)cache->_suppNormList.at(i))->getVal() : 1.0 ;
      Double_t pdfVal = pdf->getVal(nset) ;
      // Double_t pdfNorm = pdf->getNorm(nset) ;
      if (pdf->isSelectedComp()) {
	value += pdfVal*_coefCache[i]/snormVal ;
	//cout << " pdfVal = " << pdfVal << "_coefCache[" << i << "] = " << _coefCache[i] << " snormVal = " << snormVal << endl ;
//  	cxcoutD(Eval) << "RooAddPdf::evaluate(" << GetName() << ")  value += [" 
//  			<< pdf->GetName() << "] " << pdfVal << " [N= " << pdfNorm << "] * " << _coefCache[i] << " / " << snormVal << endl ;
      }
    }
    i++ ;
  }

  return value ;
}


void RooAddPdf::resetErrorCounters(Int_t resetValue)
{
  // Reset error counter to given value, limiting the number
  // of future error messages for this pdf to 'resetValue'
  RooAbsPdf::resetErrorCounters(resetValue) ;
  _coefErrCount = resetValue ;
}


Bool_t RooAddPdf::checkObservables(const RooArgSet* nset) const 
{
  // Check if PDF is valid for given normalization set.
  // Coeffient and PDF must be non-overlapping, but pdf-coefficient 
  // pairs may overlap each other

  Bool_t ret(kFALSE) ;

  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsReal* coef ;
  RooAbsReal* pdf ;
  while((coef=(RooAbsReal*)_coefIter->Next())) {
    pdf = (RooAbsReal*)_pdfIter->Next() ;
    if (pdf->observableOverlaps(nset,*coef)) {
      coutE(InputArguments) << "RooAddPdf::checkObservables(" << GetName() << "): ERROR: coefficient " << coef->GetName() 
			    << " and PDF " << pdf->GetName() << " have one or more dependents in common" << endl ;
      ret = kTRUE ;
    }
  }
  
  return ret ;
}


Int_t RooAddPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					 const RooArgSet* normSet, const char* rangeName) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code
  //
  // RooAddPdf queries each component PDF for its analytical integration capability of the requested
  // set ('allVars'). It finds the largest common set of variables that can be integrated
  // by all components. If such a set exists, it reconfirms that each component is capable of
  // analytically integrating the common set, and combines the components individual integration
  // codes into a single integration code valid for RooAddPdf.


  RooArgSet* allDepVars = getObservables(allVars) ;
  RooArgSet allAnalVars(*allDepVars) ;
  delete allDepVars ;

  TIterator* avIter = allVars.createIterator() ;

  Int_t n(0) ;

  // First iteration, determine what each component can integrate analytically
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {
    RooArgSet subAnalVars ;
    pdf->getAnalyticalIntegralWN(allVars,subAnalVars,normSet,rangeName) ;

    // Observables that cannot be integrated analytically by this component are dropped from the common list
    avIter->Reset() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)avIter->Next())) {
      if (!subAnalVars.find(arg->GetName()) && pdf->dependsOn(*arg)) {
	allAnalVars.remove(*arg,kTRUE,kTRUE) ;
      }	
    }
    n++ ;
  }

  // If no observables can be integrated analytically, return code 0 here
  if (allAnalVars.getSize()==0) {
    delete avIter ;
    return 0 ;
  }


  // Now retrieve codes for integration over common set of analytically integrable observables for each component
  _pdfIter->Reset() ;
  n=0 ;
  Int_t* subCode = new Int_t[_pdfList.getSize()] ;
  Bool_t allOK(kTRUE) ;
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {
    RooArgSet subAnalVars ;
    RooArgSet* allAnalVars2 = pdf->getObservables(allAnalVars) ;
    subCode[n] = pdf->getAnalyticalIntegralWN(*allAnalVars2,subAnalVars,normSet,rangeName) ;
    if (subCode[n]==0 && allAnalVars2->getSize()>0) {
      coutE(InputArguments) << "RooAddPdf::getAnalyticalIntegral(" << GetName() << ") WARNING: component PDF " << pdf->GetName() 
			    << "   advertises inconsistent set of integrals (e.g. (X,Y) but not X or Y individually."
			    << "   Distributed analytical integration disabled. Please fix PDF" << endl ;
      allOK = kFALSE ;
    }
    delete allAnalVars2 ; 
    n++ ;
  }  
  if (!allOK) return 0 ;

  // Mare all analytically integrated observables as such
  analVars.add(allAnalVars) ;

  // Store set of variables analytically integrated
  RooArgSet* intSet = new RooArgSet(allAnalVars) ;
  Int_t masterCode = _codeReg.store(subCode,_pdfList.getSize(),intSet)+1 ;

  delete[] subCode ;
  delete avIter ;

  return masterCode ;
}


Double_t RooAddPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const 
{
  // Return analytical integral defined by given scenario code

  // WVE needs adaptation to handle new rangeName feature

  if (code==0) {
    return getVal(normSet) ;
  }

  // Retrieve analytical integration subCodes and set of observabels integrated over
  RooArgSet* intSet ;
  const Int_t* subCode = _codeReg.retrieve(code-1,intSet) ;
  if (!subCode) {
    coutE(InputArguments) << "RooAddPdf::analyticalIntegral(" << GetName() << "): ERROR unrecognized integration code, " << code << endl ;
    assert(0) ;    
  }

  CacheElem* cache = getProjCache(normSet,intSet,0) ; // WVE rangename here?
  updateCoefficients(*cache,normSet) ;

  // Calculate the current value of this object  
  Double_t value(0) ;

  // Do running sum of coef/pdf pairs, calculate lastCoef.
  _pdfIter->Reset() ;
  _coefIter->Reset() ;
  RooAbsPdf* pdf ;
  Double_t snormVal ;
  Int_t i(0) ;

//   cout << "ROP::aIWN updateCoefCache with rangeName = " << (rangeName?rangeName:"<null>") << endl ;
  RooArgList* snormSet = (cache->_suppNormList.getSize()>0) ? &cache->_suppNormList : 0 ;
  while((pdf = (RooAbsPdf*)_pdfIter->Next())) {
    if (_coefCache[i]) {
      snormVal = snormSet ? ((RooAbsReal*) cache->_suppNormList.at(i))->getVal() : 1.0 ;
      
      // WVE swap this?
      Double_t val = pdf->analyticalIntegralWN(subCode[i],normSet,rangeName) ;
      if (pdf->isSelectedComp()) {
	
	value += val*_coefCache[i]/snormVal ;
	//  	if (_verboseEval<0) {
	// 	cout << "RAP::aI(" << GetName() << "): value += " << val << " * " << _coefCache[i] << " / " << snormVal << endl ;
	//  	}      }
      }
      i++ ;
    }    
  }

  return value ;
}




Double_t RooAddPdf::expectedEvents(const RooArgSet* nset) const 
{  
  // Return the number of expected events, which is either the sum of all coefficients
  // or the sum of the components extended terms

  Double_t expectedTotal(0.0);
  RooAbsPdf* pdf ;
    
  if (_allExtendable) {
    
    // Sum of the extended terms
    _pdfIter->Reset() ;
    while((pdf = (RooAbsPdf*)_pdfIter->Next())) {      
      expectedTotal += pdf->expectedEvents(nset) ;
    }   
    
  } else {
    
    // Sum the coefficients
    _coefIter->Reset() ;
    RooAbsReal* coef ;
    while((coef=(RooAbsReal*)_coefIter->Next())) {
      expectedTotal += coef->getVal() ;
    }   
  }

  return expectedTotal;
}


void RooAddPdf::selectNormalization(const RooArgSet* depSet, Bool_t force) 
{
  // Ignore automatic adjustments if an explicit reference normalization has been selected

  if (!force && _refCoefNorm.getSize()!=0) {
    return ;
  }

  if (!depSet) {
    fixCoefNormalization(RooArgSet()) ;
    return ;
  }

  RooArgSet* myDepSet = getObservables(depSet) ;
  fixCoefNormalization(*myDepSet) ;
  delete myDepSet ;
}


void RooAddPdf::selectNormalizationRange(const char* rangeName, Bool_t force) 
{
  // Ignore automatic adjustments if an explicit reference range has been selected
  if (!force && _refCoefRangeName) {
    return ;
  }

  fixCoefRange(rangeName) ;
}



RooAbsGenContext* RooAddPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					const RooArgSet* auxProto, Bool_t verbose) const 
{
  return new RooAddGenContext(*this,vars,prototype,auxProto,verbose) ;
}



RooArgList RooAddPdf::CacheElem::containedArgs(Action) 
{
  RooArgList allNodes;
  allNodes.add(_projList) ;
  allNodes.add(_suppProjList) ;
  allNodes.add(_refRangeProjList) ;
  allNodes.add(_rangeProjList) ;

  return allNodes ;
}

