/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
// RooProdPdf is an efficient implementation of a product of PDFs of the form 
//
//  PDF_1 * PDF_2 * ... * PDF_N
//
// RooProdPdf relies on each component PDF to be normalized and will perform no 
// explicit normalization itself. No PDF may share any dependents with any other PDF. 
//
// If exactly one of the component PDFs supports extended likelihood fits, the
// product will also be usable in extended mode, returning the number of expected
// events from the extendable component PDF. The extendable component does not
// have to appear in any specific place in the list.
// 
// To construct a product of PDFs that share dependents, and thus require explicit
// normalization of the product, use RooGenericPdf.

#include "TIterator.h"
#include "RooFitCore/RooProdPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooProdGenContext.hh"

ClassImp(RooProdPdf)
;


RooProdPdf::RooProdPdf(const char *name, const char *title, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1),
  _partListMgr(10)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		       RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1),
  _partListMgr(10)
{
  // Constructor with 2 PDFs (most frequent use case).
  // 
  // The optional cutOff parameter can be used as a speed optimization if
  // one or more of the PDF have sizable regions with very small values,
  // which would pull the entire product of PDFs to zero in those regions.
  //
  // After each PDF multiplication, the running product is compared with
  // the cutOff parameter. If the running product is smaller than the
  // cutOff value, the product series is terminated and remaining PDFs
  // are not evaluated.
  //
  // There is no magic value of the cutOff, the user should experiment
  // to find the appropriate balance between speed and precision.
  // If a cutoff is specified, the PDFs most likely to be small should
  // be put first in the product. The default cutOff value is zero.

  _pdfList.add(pdf1) ;
  if (pdf1.canBeExtended()) {
    _extendedIndex = _pdfList.index(&pdf1) ;
  }

  _pdfList.add(pdf2) ;
  if (pdf2.canBeExtended()) {
    if (_extendedIndex>=0) {
      // Protect against multiple extended terms
      cout << "RooProdPdf::RooProdPdf(" << GetName() 
	   << ") multiple components with extended terms detected,"
	   << " product will not be extendible." << endl ;
      _extendedIndex=-1 ;
    } else {
      _extendedIndex=_pdfList.index(&pdf2) ;
    }
  }
}



RooProdPdf::RooProdPdf(const char* name, const char* title, const RooArgList& pdfList, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1),
  _partListMgr(10)
{
  // Constructor from a list of PDFs
  // 
  // The optional cutOff parameter can be used as a speed optimization if
  // one or more of the PDF have sizable regions with very small values,
  // which would pull the entire product of PDFs to zero in those regions.
  //
  // After each PDF multiplication, the running product is compared with
  // the cutOff parameter. If the running product is smaller than the
  // cutOff value, the product series is terminated and remaining PDFs
  // are not evaluated.
  //
  // There is no magic value of the cutOff, the user should experiment
  // to find the appropriate balance between speed and precision.
  // If a cutoff is specified, the PDFs most likely to be small should
  // be put first in the product. The default cutOff value is zero.

  TIterator* iter = pdfList.createIterator() ;
  RooAbsPdf* pdf ;
  Int_t numExtended(0) ;
  while(pdf=(RooAbsPdf*)iter->Next()) {
    if (!dynamic_cast<RooAbsPdf*>(pdf)) {
      cout << "RooProdPdf::RooProdPdf(" << GetName() << ") list arg " 
	   << pdf->GetName() << " is not a PDF, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;
    if (pdf->canBeExtended()) {
      _extendedIndex = _pdfList.index(pdf) ;
      numExtended++ ;
    }
  }

  // Protect against multiple extended terms
  if (numExtended>1) {
    cout << "RooProdPdf::RooProdPdf(" << GetName() 
	 << ") WARNING: multiple components with extended terms detected,"
	 << " product will not be extendible." << endl ;
    _extendedIndex = -1 ;
  }

  delete iter ;
}


RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name), 
  _pdfList("_pdfList",this,other._pdfList),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(other._cutOff),
  _codeReg(other._codeReg),
  _extendedIndex(other._extendedIndex),
  _partListMgr(other._partListMgr)
{
  // Copy constructor
}


RooProdPdf::~RooProdPdf()
{
  // Destructor

  delete _pdfIter ;
}


Double_t RooProdPdf::evaluate() const 
{
  // Calculate current unnormalized value of object

  const RooArgSet* nset = _pdfList.nset() ;
  Int_t code ;
  RooArgList* plist = getPartIntList(nset,0,code) ;

  return calculate(plist,nset) ;
}



Double_t RooProdPdf::calculate(const RooArgList* partIntList, const RooArgSet* normSet) const
{
  // Calculate running product of pdfs, skipping factorized components
  RooAbsReal* partInt ;
  Double_t value(1.0) ;
  Int_t n = partIntList->getSize() ;

  Int_t i ;
  for (i=0 ; i<n ; i++) {
    partInt = ((RooAbsReal*)partIntList->at(i)) ;
    Double_t piVal = partInt->getVal(normSet) ;
    value *= piVal ;
//     if (_verboseEval<0) {
//       cout << "RPP:calc(" << GetName() << "): value *= " << piVal << " (" << partInt->GetName() << ")" << endl ;
//     }
    if (value<_cutOff) {
      //cout << "RooProdPdf::calculate(" << GetName() << ") calculation cut off after " << partInt->GetName() << endl ; 
      break ;
    }
  }

  return value ;
}


RooArgList* RooProdPdf::getPartIntList(const RooArgSet* nset, const RooArgSet* iset, Int_t& code) const
{
  // Check if this configuration was created before
  RooArgList* partIntList = _partListMgr.getNormList(this,nset,iset) ;
  if (partIntList) {
    code = _partListMgr.lastIndex() ;
    return partIntList ;
  }

  // Create the partial integral set for this request
  _pdfIter->Reset() ;
  partIntList = new RooArgList("partIntList") ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    
    // Check if all dependents of this PDF component appear in the normalization set
    Bool_t fact(kFALSE) ;
    RooArgSet *pdfDepList = pdf->getDependents(nset) ;
    if (pdfDepList->getSize()>0) {
      fact=kTRUE ;
      TIterator* depIter = pdfDepList->createIterator() ;
      RooAbsArg* dep ;
      while(dep=(RooAbsArg*)depIter->Next()) {
	if (!iset || !iset->find(dep->GetName())) {
	  fact=kFALSE ;
	}
      }
      delete depIter ;
    }
    // fact=true -> all integrated pdf dependents are in normalization set

    if (fact) {
      // This product term factorizes, no partial integral needs to be created
    } else if (nset && pdfDepList->getSize()==0) {
    } else {
      if (iset && iset->getSize()>0) {
	RooArgSet* iSet = pdf->getDependents(iset) ;
	RooAbsReal* partInt = pdf->createIntegral(*iSet,*pdfDepList) ;
	partInt->setOperMode(operMode()) ;
	partIntList->addOwned(*partInt) ;
	delete iSet ;
      } else {
	partIntList->add(*pdf) ;
      }
    }

    delete pdfDepList ;
  }

  // Store the partial integral list and return the assigned code ;
  code = _partListMgr.setNormList(this,nset,iset,partIntList) ;
//   cout << "RooProdPdf::getPIL(" << GetName() << ") creating new configuration with code " << code << endl
//        << "    nset = " ; if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
//   cout << "    iset = " ; if (iset) iset->Print("1") ; else cout << "<none>" << endl ;
//   cout << "    Partial Integral List:" << endl ;
//   partIntList->Print("1") ;

  return partIntList ;
}


Int_t RooProdPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code.
  //
  // RooProdPdf implements two strategies in implementing analytical integrals
  //
  // First, PDF components whose entire set of dependents are requested to be integrated
  // can be dropped from the product, as they will integrate out to 1 by construction
  //
  // Second, RooProdPdf queries each remaining component PDF for its analytical integration 
  // capability of the requested set ('allVars'). It finds the largest common set of variables 
  // that can be integrated by all remaining components. If such a set exists, it reconfirms that 
  // each component is capable of analytically integrating the common set, and combines the components 
  // individual integration codes into a single integration code valid for RooProdPdf.

  if (_forceNumInt) return 0 ;
  
  // Declare that we can analytically integrate all requested observables
  analVars.add(allVars) ;

  // Retrieve (or create) the required partial integral list
  Int_t code ;
  getPartIntList(normSet,&allVars,code) ;
  
  return code+1 ;
}



Double_t RooProdPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
{
  // Return analytical integral defined by given scenario code

  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // Partial integration scenarios
  RooArgList* partIntList = _partListMgr.getNormListByIndex(code-1) ;
  return calculate(partIntList,normSet) ;
}



Bool_t RooProdPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check that none of the PDFs have overlapping dependents
  
  Bool_t ret(kFALSE) ;
  
  _pdfIter->Reset() ;
  RooAbsPdf* pdf, *pdf2 ;
  TIterator* iter2 = _pdfList.createIterator() ;
  while(pdf = (RooAbsPdf*)_pdfIter->Next()) {
    *iter2 = *_pdfIter ;
    while(pdf2 = (RooAbsPdf*)iter2->Next()) {
      if (pdf->dependentOverlaps(nset,*pdf2)) {
	cout << "RooProdPdf::checkDependents(" << GetName() << "): ERROR: PDFs " << pdf->GetName() 
	     << " and " << pdf2->GetName() << " have one or more dependents in common" << endl ;
	ret = kTRUE ;
      }    
    }
  }
  delete iter2 ;
  return ret ;
}




RooAbsPdf::ExtendMode RooProdPdf::extendMode() const
{
  return (_extendedIndex>=0) ? ((RooAbsPdf*)_pdfList.at(_extendedIndex))->extendMode() : CanNotBeExtended ;
}



Double_t RooProdPdf::expectedEvents() const 
{
  assert(_extendedIndex>=0) ;
  return ((RooAbsPdf*)_pdfList.at(_extendedIndex))->expectedEvents() ;
}




RooAbsGenContext* RooProdPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, Bool_t verbose) const 
{
  return new RooProdGenContext(*this,vars,prototype,verbose) ;
}




void RooProdPdf::operModeHook() 
{
  Int_t i ;
  for (i=0 ; i<_partListMgr.cacheSize() ; i++) {
    RooArgList* plist = _partListMgr.getNormListByIndex(i) ;
   if (plist->isOwning()) {
     TIterator* iter = plist->createIterator() ;
     RooAbsArg* arg ;
     while(arg=(RooAbsArg*)iter->Next()) {
       arg->setOperMode(_operMode) ;
     }
     delete iter ;
   }
  }
  return ;
}



Bool_t RooProdPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  Bool_t ret(kFALSE) ;  

  Int_t i ;
  for (i=0 ; i<_partListMgr.cacheSize() ; i++) {
    RooArgList* plist = _partListMgr.getNormListByIndex(i) ;
    TIterator* iter = plist->createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
    }
    delete iter ;
  }
  return ret ;
}
