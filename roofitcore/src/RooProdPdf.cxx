/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.cc,v 1.24 2002/06/03 22:15:53 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   06-Jan-2000 DK Created initial version
 *   19-Apr-2000 DK Add the printEventStats() method
 *   26-Jun-2000 DK Add support for extended likelihood fits
 *   02-Jul-2000 DK Add support for multiple terms (instead of only 2)
 *   05-Jul-2000 DK Add support for extended maximum likelihood and a
 *                  new method for this: setNPar()
 *   03-May02001 WV Port to RooFitCore/RooFitModels
 *
 * Copyright (C) 2000 Stanford University
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
  _pdfIter(_pdfList.createIterator()), 
//   _partIntSet1("partIntSet1","Primary set of partial integrals",this,kFALSE,kFALSE),
//   _partIntSet2("partIntSet2","Alternate set of partial integrals",this,kFALSE,kFALSE),
  _intIter1(_partIntSet1.createIterator()),
  _intIter2(_partIntSet2.createIterator()),
  _lastAICode1(-1),
  _lastAICode2(-1),
  _intIter(0),  
  _nextSet(1),
  _lastEvalNSet((RooArgSet*)-1),
  _evalCode(0),
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		       RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
//   _partIntSet1("partIntSet1","Primary set of partial integrals",this,kFALSE,kFALSE),
//   _partIntSet2("partIntSet2","Alternate set of partial integrals",this,kFALSE,kFALSE),
  _intIter1(_partIntSet1.createIterator()),
  _intIter2(_partIntSet2.createIterator()),
  _lastAICode1(-1),
  _lastAICode2(-1),
  _intIter(0),
  _nextSet(1),
  _lastEvalNSet((RooArgSet*)-1),
  _evalCode(0),
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1)
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
//   _partIntSet1("partIntSet1","Primary set of partial integrals",this,kFALSE,kFALSE),
//   _partIntSet2("partIntSet2","Alternate set of partial integrals",this,kFALSE,kFALSE),
  _intIter1(_partIntSet1.createIterator()),
  _intIter2(_partIntSet2.createIterator()),
  _lastAICode1(-1),
  _lastAICode2(-1),
  _intIter(0),
  _nextSet(1),
  _lastEvalNSet((RooArgSet*)-1),
  _evalCode(0),
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1)
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
//   _partIntSet1("partIntSet1","Primary set of partial integrals",this,kFALSE,kFALSE),
//   _partIntSet2("partIntSet2","Alternate set of partial integrals",this,kFALSE,kFALSE),
  _intIter1(_partIntSet1.createIterator()),
  _intIter2(_partIntSet2.createIterator()),
  _lastAICode1(-1),
  _lastAICode2(-1),
  _intIter(0),
  _nextSet(1),
  _lastEvalNSet((RooArgSet*)-1),
  _evalCode(0),
  _cutOff(other._cutOff),
  _codeReg(other._codeReg),
  _extendedIndex(other._extendedIndex)
{
  // Copy constructor
}


RooProdPdf::~RooProdPdf()
{
  // Destructor

  delete _pdfIter ;
  delete _intIter ;
}


Double_t RooProdPdf::evaluate() const 
{
  // Calculate current unnormalized value of object

  const RooArgSet* nset = _pdfList.nset() ;
  if (nset != _lastEvalNSet) {
    RooArgSet allVars ;
    RooArgSet analVars ;
    _evalCode = getAnalyticalIntegralWN(allVars,analVars,nset) ;
    _lastEvalNSet = (RooArgSet*) nset ;
    //cout << "RooProdPdf::evaluate(" << GetName() << "): new code = " << _evalCode << endl ;    
  }

  return analyticalIntegralWN(_evalCode,nset) ;
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
	cout << "RooAddPdf::checkDependents(" << GetName() << "): ERROR: PDFs " << pdf->GetName() 
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

//   cout << "RooProdPdf::getAIWN(" << GetName() << ")" << endl 
//        << "                   allVars = " ; allVars.Print("1") ;
//   cout << "                   normSet = " ; if (normSet) normSet->Print("1") ; else cout << "<none>" << endl ;
  
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Int_t code(0), n(0) ;
  Bool_t allFact(kTRUE) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {

    // Check if all dependents of this PDF component appear in the normalization set
    Bool_t fact(kFALSE) ;
    RooArgSet *pdfDepList = pdf->getDependents(normSet) ;
    if (pdfDepList->getSize()>0) {
      fact=kTRUE ;
      TIterator* depIter = pdfDepList->createIterator() ;
      RooAbsArg* dep ;
      while(dep=(RooAbsArg*)depIter->Next()) {
	if (!allVars.find(dep->GetName())) {
	  fact=kFALSE ;
	}
      }
    }
    if (!fact) allFact=kFALSE ;
    delete pdfDepList ;
  }


  analVars.add(allVars) ;

  if (allFact) {
    // Fully factorizing integral has special code 1000
    return 1000 ;
  } else {
    // Partial integral product. Code assigned by registry
    Int_t code(2) ;
    RooArgSet* iSet = (RooArgSet*) analVars.snapshot(kFALSE) ;
    RooArgSet* nSet = normSet ? ((RooArgSet*)normSet->snapshot(kFALSE)) : 0 ;
    Int_t masterCode = _codeReg.store(&code,1,iSet,nSet)+1 ;    
    return masterCode ;
  }
  return 0 ;
}


void RooProdPdf::syncAnaInt(Int_t code) const
{  
  // Do we need to do anything?

  if (code == _lastAICode1) {
    _intIter = _intIter1 ;
    return ;
  }

  if (code == _lastAICode2) {
    _intIter = _intIter2 ;
    return ;
  }

  if (_nextSet==1) {
    syncAnaInt(_partIntSet1,code) ;
    _lastAICode1 = code ;
    _nextSet = 2 ;
    _intIter = _intIter1 ;
  } else {
    syncAnaInt(_partIntSet2,code) ;
    _lastAICode2 = code ;
    _nextSet = 1;
    _intIter = _intIter2 ;
  }
}



void RooProdPdf::syncAnaInt(RooArgSet& partIntSet, Int_t code) const
{
  partIntSet.removeAll() ;

  // Retrieve information from registry
  RooArgSet* intSet ;
  RooArgSet* normSet ;
  _codeReg.retrieve(code-1,intSet,normSet) ;

//   cout << "RooProdPdf::syncAnaInt code = " << code << endl;
//   cout << "intSet  = " ; intSet->Print("1") ;
//   cout << "normSet = " ; if (normSet) normSet->Print("1") ; else cout << "<none>" << endl ;

  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Bool_t allFact(kTRUE) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {

    // Check if all dependents of this PDF component appear in the normalization set
    Bool_t fact(kFALSE) ;
    RooArgSet *pdfDepList = pdf->getDependents(normSet) ;
    if (pdfDepList->getSize()>0) {
      fact=kTRUE ;
      TIterator* depIter = pdfDepList->createIterator() ;
      RooAbsArg* dep ;
      while(dep=(RooAbsArg*)depIter->Next()) {
	if (!intSet->find(dep->GetName())) {
	  fact=kFALSE ;
	}
      }
    }
    if (!fact) allFact=kFALSE ;

    // fact = true if all pdf dependents are in normalization set

    if (fact) {
      // This product term factorizes, no partial integral needs to be created
    } else if (normSet && pdfDepList->getSize()==0) {
    } else {
      if (intSet->getSize()>0) {
	RooArgSet* iSet = pdf->getDependents(intSet) ;
	RooAbsReal* partInt = pdf->createIntegral(*iSet,*pdfDepList) ;
	partInt->setOperMode(operMode()) ;
	partIntSet.addOwned(*partInt) ;
	delete iSet ;
      } else {
	partIntSet.add(*pdf) ;
      }
    }

    delete pdfDepList ;
  }
}


void RooProdPdf::operModeHook() 
{
  if (_partIntSet1.isOwning()) {
    _intIter1->Reset() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)_intIter1->Next()) {
      arg->setOperMode(_operMode) ;
    }
  }
  if (_partIntSet2.isOwning()) {
    _intIter2->Reset() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)_intIter2->Next()) {
      arg->setOperMode(_operMode) ;
    }
  }
  return ;
}


Double_t RooProdPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const 
{
  // Return analytical integral defined by given scenario code

  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // Full normalized integration scenario
  if (code==1000) {
    return 1.0 ;
  }

  // Partial integration scenarios
  syncAnaInt(code) ;
  
  // Calculate running product of pdfs, skipping factorized components
  RooAbsReal* partInt ;
  Double_t value(1.0) ;
  _intIter->Reset() ;

  while(partInt=(RooAbsReal*)_intIter->Next()) {    
    Double_t piVal = partInt->getVal(normSet) ;
    value *= piVal ;
    //cout << GetName() << ": value *= " << piVal << " (" << partInt->GetName() << ")" << endl ;
    if (value<_cutOff) {
      //cout << "RooProdPdf::aIWN(" << GetName() << ") calculation cut off after " << partInt->GetName() << endl ; 
      break ;
    }
  }

  return value ;
}


RooAbsGenContext* RooProdPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, Bool_t verbose) const 
{
  return new RooProdGenContext(*this,vars,prototype,verbose) ;
}


Bool_t RooProdPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  Bool_t ret(kFALSE) ;  

  _intIter1->Reset() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)_intIter1->Next()) {
    ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
  }

  _intIter2->Reset() ;
  while(arg=(RooAbsArg*)_intIter2->Next()) {
    ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
  }

  return ret ;
}
