/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdPdf.cc,v 1.41 2003/07/30 01:19:39 wverkerke Exp $
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
#include "RooFitCore/RooGenProdProj.hh"

ClassImp(RooProdPdf)
;


RooProdPdf::RooProdPdf(const char *name, const char *title, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _partListMgr(10),
  _partOwnedListMgr(10),
  _genCode(10),
  _cutOff(cutOff),
  _pdfList("_pdfList","List of PDFs",this),
  _extendedIndex(-1),
  _useDefaultGen(kFALSE)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		       RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), 
  _partListMgr(10),
  _partOwnedListMgr(10),
  _genCode(10),
  _cutOff(cutOff),
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _extendedIndex(-1),
  _useDefaultGen(kFALSE)
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
  _partListMgr(10),
  _partOwnedListMgr(10),
  _genCode(10),
  _cutOff(cutOff),
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _extendedIndex(-1),
  _useDefaultGen(kFALSE)
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
  _partListMgr(other._partListMgr),
  _partOwnedListMgr(other._partOwnedListMgr),
  _genCode(other._genCode),
  _cutOff(other._cutOff),
  _pdfList("_pdfList",this,other._pdfList),
  _pdfIter(_pdfList.createIterator()), 
  _extendedIndex(other._extendedIndex),
  _useDefaultGen(other._useDefaultGen) 
{
  // Copy constructor
}


RooProdPdf::~RooProdPdf()
{
  // Destructor

  delete _pdfIter ;
}


Double_t RooProdPdf::getVal(const RooArgSet* set) const 
{
  _curNormSet = (RooArgSet*)set ;
  return RooAbsPdf::getVal(set) ;
}


Double_t RooProdPdf::evaluate() const 
{
  // Calculate current unnormalized value of object

  // WVE NSET may not exists, but getPartIntList may need it...

  Int_t code ;
  RooArgList* plist = getPartIntList(_curNormSet,0,code) ;

  return calculate(plist,_curNormSet) ;
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



TList* RooProdPdf::factorizeProduct(const RooArgSet& normSet) const 
{
  // Factorize product in irreducible terms for given choice of integration/normalization

//   cout << "RooProdPdf::factorizeProduct(" << GetName() << ")" << endl ;
//   cout << "   normSet  = " ; normSet.Print("1") ;
 
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;

  // Setup lists for factorization terms and their dependents
  TList* list = new TList ;
  TList dlist ;
  RooArgSet* term ;
  RooArgSet* termDeps ;
  TIterator* lIter = list->MakeIterator() ;
  TIterator* ldIter = dlist.MakeIterator() ;

  // Loop over the PDFs
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    lIter->Reset() ;
    ldIter->Reset() ;

    // Check if this PDF has dependents overlapping with one of the existing terms
    Bool_t done(kFALSE) ;
    while(term=(RooArgSet*)lIter->Next()) {      
      termDeps=(RooArgSet*)ldIter->Next() ;

      if (pdf->dependsOn(*termDeps)) {
	term->add(*pdf) ;
	RooArgSet* deps = pdf->getDependents(normSet) ;
	termDeps->add(*deps,kFALSE) ;
	delete deps ;
	done = kTRUE ;
      }
    }

    // If not, create a new term
    if (!done) {
      term = new RooArgSet ;
      termDeps = new RooArgSet ;
      term->add(*pdf) ;
      RooArgSet* deps = pdf->getDependents(normSet) ;
      termDeps->add(*deps,kFALSE) ;
      delete deps ;
      list->Add(term) ;
      dlist.Add(termDeps) ;
    }
  }

  lIter->Reset() ;
//   cout << "list of terms:" << endl ;
//   while(term=(RooArgSet*)lIter->Next()) {
//     term->Print("1") ;
//   }

  delete lIter ;
  delete ldIter ;
  dlist.Delete() ;

  return list ;
}




RooArgList* RooProdPdf::getPartIntList(const RooArgSet* nset, const RooArgSet* iset, Int_t& code) const
{
  // Check if this configuration was created before
  RooArgList* partIntList = _partListMgr.getNormList(this,nset,iset) ;
  if (partIntList) {
    code = _partListMgr.lastIndex() ;
    return partIntList ;
  }

  // WVE --- NSET object may not exist?

  // Factorize the product in irreducible terms for this nset
  TList* terms = factorizeProduct(nset?(*nset):RooArgSet()) ;

  // Iterate over the irreducible terms to create normalized projection integrals for them
  TIterator* tIter = terms->MakeIterator() ;
  partIntList = new RooArgList("partIntList") ;
  RooArgList* partIntOwnedList = new RooArgList("partIntList") ;

  RooArgSet* term ;
  
  while(term=(RooArgSet*)tIter->Next()) {

    RooArgSet termNSet, termISet ;

    // Make the list of dependents and integrated dependents for this term
    TIterator* pIter = term->createIterator() ;
    RooAbsPdf* pdf ;
    while(pdf=(RooAbsPdf*)pIter->Next()) {
      if (nset) {
	RooArgSet* tmp = pdf->getDependents(*nset) ;
	termNSet.add(*tmp) ;
	delete tmp ;
      }
      if (iset) {
	RooArgSet* tmp = pdf->getDependents(*iset) ;
	termISet.add(*tmp) ;
	delete tmp ;
      }
    }
    delete pIter ;

    // Check if all observbales of this term are integrated. If so the term cancels
    if (termNSet.getSize()>0 && termNSet.getSize()==termISet.getSize()) {
      // Term factorizes
      continue ;
    }

    if (nset && termNSet.getSize()==0) {
      // Term needs no integration
      continue ;
    }

    if (iset && iset->getSize()>0) {
      if (term->getSize()==1) {
	// Single term needs normalized integration

	pIter = term->createIterator() ;
	pdf = (RooAbsPdf*) pIter->Next() ;
	delete pIter ;

	RooAbsReal* partInt = pdf->createIntegral(termISet,termNSet) ;
	partInt->setOperMode(operMode()) ;
	partIntList->add(*partInt) ;
	partIntOwnedList->addOwned(*partInt) ;
	continue ;

      } else {
	// Composite term needs normalized integration

	const char* name = makeRGPPName("PROJ_",*term,termISet,termNSet) ;
	RooAbsReal* partInt = new RooGenProdProj(name,name,*term,termISet,termNSet) ;
	partInt->setOperMode(operMode()) ;
	partIntList->add(*partInt) ;
	partIntOwnedList->addOwned(*partInt) ;
	continue ;

      }      
    }

    if (nset && nset->getSize()>0 && term->getSize()>1) {
      // Composite term needs normalized integration
      const char* name = makeRGPPName("PROJ_",*term,termISet,termNSet) ;
      RooAbsReal* partInt = new RooGenProdProj(name,name,*term,termISet,termNSet) ;
      partInt->setOperMode(operMode()) ;
      partIntList->add(*partInt) ;
      partIntOwnedList->addOwned(*partInt) ;
      continue ;
    }

    // Add pdfs in term straight    
    pIter = term->createIterator() ;
    while(pdf=(RooAbsPdf*)pIter->Next()) {
      partIntList->add(*pdf) ;
    }
    delete pIter ;

  }

  // Store the partial integral list and return the assigned code ;
  code = _partListMgr.setNormList(this,nset,iset,partIntList) ;
  Int_t code2 = _partOwnedListMgr.setNormList(this,nset,iset,partIntOwnedList) ;

//    cout << "RooProdPdf::getPIL(" << GetName() << "," << this << ") creating new configuration with code " << code << endl
//         << "    nset = " ; if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
//    cout << "    iset = " ; if (iset) iset->Print("1") ; else cout << "<none>" << endl ;
//    cout << "    Partial Integral List:" << endl ;
//    partIntList->Print("1") ;
//    cout << "    Partial Owned Integral List:" << endl ;
//    partIntOwnedList->Print("1") ;
//    cout << endl  ;

  return partIntList ;
}





const char* RooProdPdf::makeRGPPName(const char* pfx, const RooArgSet& term, const RooArgSet& iset, const RooArgSet& nset) const
{
  // Make an appropriate name for a RooGenProdProj object in getPartIntList() 

  static TString pname ;
  pname = pfx ;

  TIterator* pIter = term.createIterator() ;
  
  // Encode component names
  Bool_t first(kTRUE) ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)pIter->Next()) {
    if (first) {
      first = kFALSE ;
    } else {
      pname.Append("_X_") ;
    }
    pname.Append(pdf->GetName()) ;
  }
  delete pIter ;

  if (iset.getSize()>0) {
    pname.Append("_Int[") ;
    TIterator* iter = iset.createIterator() ;
    RooAbsArg* arg ;
    Bool_t first(kTRUE) ;
    while(arg=(RooAbsArg*)iter->Next()) {
      if (first) {
	first=kFALSE ;
      } else {
	pname.Append(",") ;
      }
      pname.Append(arg->GetName()) ;
    }
    delete iter ;
    pname.Append("]") ;
  }

  if (nset.getSize()>0) {
    pname.Append("_Norm[") ;
    Bool_t first(kTRUE); 
    TIterator* iter  = nset.createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      if (first) {
	first=kFALSE ;
      } else {
	pname.Append(",") ;
      }
      pname.Append(arg->GetName()) ;
    }
    delete iter ;
    pname.Append("]") ;
  }

  return pname.Data() ;
}


Bool_t RooProdPdf::forceAnalyticalInt(const RooAbsArg& dep) const 
{
  return kTRUE ;
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
  Double_t val = calculate(partIntList,normSet) ;
  
  //cout << "RPP::aIWN(" << GetName() << ") value = " << val << endl ;
  return val ;
}



Bool_t RooProdPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check that none of the PDFs have overlapping dependents
  return kFALSE ;
  
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
  if (_useDefaultGen) return RooAbsPdf::genContext(vars,prototype,verbose) ;
  return new RooProdGenContext(*this,vars,prototype,verbose) ;
}



Int_t RooProdPdf::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (!_useDefaultGen) return 0 ;


  // Find the subset directVars that only depend on a single PDF in the product
  RooArgSet directSafe ;
  TIterator* dIter = directVars.createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)dIter->Next()) {
    if (isDirectGenSafe(*arg)) directSafe.add(*arg) ;
  }
  delete dIter ;


  // Now find direct integrator for relevant components ;
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Int_t code[64], n(0) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    RooArgSet pdfDirect ;
    code[n] = pdf->getGenerator(directSafe,pdfDirect,staticInitOK) ;
    if (code[n]!=0) {
      generateVars.add(pdfDirect) ;
    }
    n++ ;
  }


  if (generateVars.getSize()>0) {
    Int_t masterCode = _genCode.store(code,n) ;
    return masterCode+1 ;    
  } else {
    return 0 ;
  }
}


void RooProdPdf::initGenerator(Int_t code)
{
  if (!_useDefaultGen) return ;

  const Int_t* codeList = _genCode.retrieve(code-1) ;
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Int_t i(0) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    if (codeList[i]!=0) {
      pdf->initGenerator(codeList[i]) ;
    }
    i++ ;
  }
}


void RooProdPdf::generateEvent(Int_t code)
{  
  if (!_useDefaultGen) return ;

  const Int_t* codeList = _genCode.retrieve(code-1) ;
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Int_t i(0) ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    if (codeList[i]!=0) {
      pdf->generateEvent(codeList[i]) ;
    }
    i++ ;
  }

}




void RooProdPdf::operModeHook() 
{
  Int_t i ;
  for (i=0 ; i<_partListMgr.cacheSize() ; i++) {
    RooArgList* plist = _partOwnedListMgr.getNormListByIndex(i) ;
    TIterator* iter = plist->createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      arg->setOperMode(_operMode) ;
    }
    delete iter ;
  }
  return ;
}



Bool_t RooProdPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) 
{
  Bool_t ret(kFALSE) ;  

//    cout << "RooPrdoPdf::redirectServersHook(" << this << ") recursive = " << (isRecursive?"T":"F") << " newServerList = " ;
//    newServerList.Print("1") ;

  Int_t i ;
  for (i=0 ; i<_partListMgr.cacheSize() ; i++) {
    RooArgList* plist = _partListMgr.getNormListByIndex(i) ;    

      // Update non-owning lists
      TIterator* iter = plist->createIterator() ;
      RooAbsArg* arg ;
      while(arg=(RooAbsArg*)iter->Next()) {
	RooAbsArg* newArg = arg->findNewServer(newServerList,nameChange) ;
	if (newArg) {
//  	  cout << "replacing server " << arg->GetName() << "(" << arg << ") with (" << newArg << ") in partList[" << i << "]" << endl ;
	  plist->replace(*arg,*newArg) ;
	}
      }
      delete iter ;      
    }



  if (isRecursive) {
    for (i=0 ; i<_partOwnedListMgr.cacheSize() ; i++) {
      RooArgList* plist = _partOwnedListMgr.getNormListByIndex(i) ;    
      
      // Forward recurive redirection calls for owning lists
      // Only redirect links to component PDFs, recursive link direction
      // of servers of PDF is handled via the regular channels
      
      TIterator* iter = plist->createIterator() ;
      RooAbsArg* arg ;
      RooAbsCollection* newPdfServerList = newServerList.selectCommon(_pdfList) ;
      while(arg=(RooAbsArg*)iter->Next()) {

//  	cout << "recursivedRed newServerList on " << arg ->GetName() << "(" << arg << ") in partOwnedList[" << i << "]" << endl ;
	ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;

      }
      delete iter ;
      delete newPdfServerList ;      
    }
  }
  

  return ret ;
}

void RooProdPdf::printCompactTreeHook(const char* indent) 
{
  Int_t i ;
  cout << indent << "RooProdPdf begin partial integral cache" << endl ;

  for (i=0 ; i<_partListMgr.cacheSize() ; i++) {
    RooArgList* plist = _partListMgr.getNormListByIndex(i) ;    

    TIterator* iter = plist->createIterator() ;
    RooAbsArg* arg ;
    TString indent2(indent) ;
    indent2 += Form("[%d] ",i) ;
    while(arg=(RooAbsArg*)iter->Next()) {      
      arg->printCompactTree(indent2) ;
    }
    delete iter ;
  }

  cout << indent << "RooProdPdf end partial integral cache" << endl ;
}



Bool_t RooProdPdf::isDirectGenSafe(const RooAbsArg& arg) const 
{
  // Only override base class behaviour if default generator method is enabled
  if (!_useDefaultGen) return RooAbsPdf::isDirectGenSafe(arg) ;

  // Argument may appear in only one PDF component
  _pdfIter->Reset() ;
  RooAbsPdf* pdf, *thePdf(0) ;  
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {

    if (pdf->dependsOn(arg)) {
      // Found PDF depending on arg

      // If multiple PDFs depend on arg directGen is not safe
      if (thePdf) return kFALSE ;

      thePdf = pdf ;
    }
  }
  // Forward call to relevant component PDF
  return thePdf?(thePdf->isDirectGenSafe(arg)):kFALSE ;
}
