/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooProdPdf.cc,v 1.26 2002/08/21 23:06:28 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   06-Aug-2002 WV Rewrite using normalization manager
 *
 * Copyright (C) 2002 University of California
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
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooSuperCategory.hh"

ClassImp(RooProdPdf)
;


RooProdPdf::RooProdPdf(const char *name, const char *title, Double_t cutOff) :
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _partList("_partList","List of partial PDF integrals",this),
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1),
  _part1ListMgr(10),
  _part2ListMgr(10),
  _cod1ListMgr(10),
  _cod2ListMgr(10)
{
  // Dummy constructor
}


RooProdPdf::RooProdPdf(const char *name, const char *title,
		       RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff) : 
  RooAbsPdf(name,title), 
  _pdfList("_pdfList","List of PDFs",this),
  _partList("_partList","List of partial PDF integrals",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1),
  _part1ListMgr(10),
  _part2ListMgr(10),
  _cod1ListMgr(10),
  _cod2ListMgr(10)
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
  _partList("_partList","List of partial PDF integrals",this),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(cutOff),
  _codeReg(10),
  _extendedIndex(-1),
  _part1ListMgr(10),
  _part2ListMgr(10),
  _cod1ListMgr(10),
  _cod2ListMgr(10)
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
  _partList("_partList",this,other._partList),
  _pdfIter(_pdfList.createIterator()), 
  _cutOff(other._cutOff),
  _codeReg(other._codeReg),
  _extendedIndex(other._extendedIndex),
  _part1ListMgr(other._part1ListMgr),
  _part2ListMgr(other._part2ListMgr),
  _cod1ListMgr(other._cod1ListMgr),
  _cod2ListMgr(other._cod2ListMgr)
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
  RooArgList* p1list, *p2list, *c1list, *c2list ;
  getPartIntList(nset,0,code,p1list,p2list,c1list,c2list) ;

  return calculate(p1list,c1list)/calculate(p2list,c2list) ;
}



Double_t RooProdPdf::calculate(const RooArgList* partIntList, const RooArgList* codList) const
{
  // Sum over all states of codList
  if (codList->getSize()==0) {
    return calculate(partIntList) ;
  }

  cout << "RPP::calc(" << GetName() << ") summing required" << endl ;
  Double_t value(0) ;
  RooSuperCategory superCat("superCat","superCat",*codList) ;
  Int_t saveState = superCat.getIndex() ;
  TIterator* iter = superCat.typeIterator() ;
  RooCatType* type ;
  while(type=(RooCatType*)iter->Next()) {
    superCat.setIndex(type->getVal()) ;
    Double_t tmp = calculate(partIntList) ;
    value += tmp ;
    cout << "value += " << tmp << endl ;
  }
  delete iter ;
  superCat.setIndex(saveState) ;
  
  return value ;
}



Double_t RooProdPdf::calculate(const RooArgList* partIntList) const
{
  // Calculate running product of pdfs, skipping factorized components
  RooAbsReal* partInt ;
  Double_t value(1.0) ;

  Int_t n = partIntList->getSize() ;
  Int_t i ;
  for (i=0 ; i<n ; i++) {
    partInt = ((RooAbsReal*)partIntList->at(i)) ;
    Double_t piVal = partInt->getVal() ;
    value *= piVal ;
    if (value<_cutOff) {
      //cout << "RooProdPdf::calculate(" << GetName() << ") calculation cut off after " << partInt->GetName() << endl ; 
      break ;
    }
  }

  return value ;
}


void RooProdPdf::getPartIntList(const RooArgSet* nset, const RooArgSet* iset, 
				Int_t& code, RooArgList*& part1List, RooArgList*& part2List,
				RooArgList*& cod1List, RooArgList*& cod2List) const
{
  // Check if this configuration was created before
  RooArgList* part1IntList = _part1ListMgr.getNormList(this,nset,iset) ;
  RooArgList* part2IntList = _part2ListMgr.getNormList(this,nset,iset) ;
  RooArgList* cod1IntList  = _cod1ListMgr.getNormList(this,nset,iset) ;
  RooArgList* cod2IntList  = _cod2ListMgr.getNormList(this,nset,iset) ;
  if (part1IntList && part2IntList) {
    code = _part1ListMgr.lastIndex() ;
    part1List = part1IntList ;
    part2List = part2IntList ;
    cod1List  = cod1IntList ;
    cod2List  = cod2IntList ;
    return  ;
  }

  // Create the partial integral set for this request
  part1IntList = new RooArgList("part1IntList") ;
  part2IntList = new RooArgList("part2IntList") ;
  cod1IntList  = getCategoryOverlapDeps(iset) ; 
  cod2IntList  = getCategoryOverlapDeps(nset) ; 

  RooArgSet* iset2 = 0, *nset2 = 0 ;
  if (iset) {
    iset2 = new RooArgSet(*iset) ;
    iset2->remove(*cod1IntList,kTRUE,kTRUE) ;
  }
  if (nset) {
    nset2 = new RooArgSet(*nset) ;
    nset2->remove(*cod2IntList,kTRUE,kTRUE) ;
  }
  

  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  while(pdf=(RooAbsPdf*)_pdfIter->Next()) {
    
    // Check if all dependents of this PDF component appear in the normalization set
    Bool_t fact(kFALSE) ;
    RooArgSet *pdfDepList = pdf->getDependents(nset2) ;
    if (pdfDepList->getSize()>0) {
      fact=kTRUE ;
      TIterator* depIter = pdfDepList->createIterator() ;
      RooAbsArg* dep ;
      while(dep=(RooAbsArg*)depIter->Next()) {
	if (!iset2 || !iset2->find(dep->GetName())) {
	  fact=kFALSE ;
	}
      }
      delete depIter ;
    }
    // fact=true -> all integrated pdf dependents are in normalization set

    if (fact) {
      // This product term factorizes, no partial integral needs to be created
    } else if (nset2 && pdfDepList->getSize()==0) {
    } else {
      if (iset2 && iset2->getSize()>0) {
	RooArgSet* iSet = pdf->getDependents(iset2) ;
	RooAbsReal* part1Int = pdf->createIntegral(*iSet) ;
	part1IntList->addOwned(*part1Int) ;

	if (nset2) {
	  RooAbsReal* part2Int = pdf->createIntegral(*pdfDepList) ;
	  part2IntList->addOwned(*part2Int) ;
	}

	delete iSet ;
      } else {

	part1IntList->add(*pdf) ;

	RooAbsReal* part2Int = pdf->createIntegral(*pdfDepList) ;
	part2Int->setOperMode(operMode()) ;
	part2IntList->addOwned(*part2Int) ;

      }
    }

    delete pdfDepList ;
  }

  delete iset2 ;
  delete nset2 ;

  // Store the partial integral list and return the assigned code ;
  code = _part1ListMgr.setNormList(this,nset,iset,part1IntList) ;
  code = _part2ListMgr.setNormList(this,nset,iset,part2IntList) ;
  code = _cod1ListMgr.setNormList(this,nset,iset,cod1IntList) ;
  code = _cod2ListMgr.setNormList(this,nset,iset,cod2IntList) ;

//    cout << "RooProdPdf::getPIL(" << GetName() << ") creating new configuration with code " << code << endl
//         << "    nset = " ; if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
//    cout << "    iset = " ; if (iset) iset->Print("1") ; else cout << "<none>" << endl ;
//    cout << "    Partial Integral List 1 :" ; part1IntList->Print("1") ;
//    cout << "    Partial Integral List 2 :" ; part2IntList->Print("1") ;
//    cout << "    Category Overlap Dependents List 1 :" ; cod1IntList->Print("1") ;
//    cout << "    Category Overlap Dependents List 2 :" ;  cod2IntList->Print("1") ;
 
   
  part1List = part1IntList ;
  part2List = part2IntList ;
  cod1List = cod1IntList ;
  cod2List = cod2IntList ;

  _partList.removeAll() ;
  _partList.add(*part1IntList) ;
  _partList.add(*part2IntList) ;
  return ;
}



RooArgList* RooProdPdf::getCategoryOverlapDeps(const RooArgSet* depSet) const 
{
  RooArgList* catDepsO = new RooArgList("codList") ;

  // Reduce nset to real-valued dependents only. Category dependent overlaps are allowed
  if (!depSet) return catDepsO ;

  RooArgSet catDeps ;
  TIterator* iter = depSet->createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (dynamic_cast<RooAbsCategory*>(arg)) {
      catDeps.add(*arg) ;
    }
  }
  delete iter ;
  
  _pdfIter->Reset() ;
  RooAbsPdf* pdf, *pdf2 ;
  TIterator* iter2 = _pdfList.createIterator() ;
  while(pdf = (RooAbsPdf*)_pdfIter->Next()) {
    RooArgSet* pdf1Deps = pdf->getDependents(catDeps) ;
    *iter2 = *_pdfIter ;
    while(pdf2 = (RooAbsPdf*)iter2->Next()) {
      RooArgSet* pdf2Deps = pdf2->getDependents(catDeps) ;

      RooArgSet* common = (RooArgSet*) pdf1Deps->selectCommon(*pdf2Deps) ;
      if (common->getSize()>0) {
	catDepsO->addClone(*common,kTRUE) ;
      }

      delete pdf2Deps ;
    }
    delete pdf1Deps ;
  }
  delete iter2 ;

  return catDepsO ;
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
  RooArgList* p1list, *p2list, *c1list, *c2list ;
  getPartIntList(normSet,&allVars,code,p1list,p2list,c1list,c2list) ;
  
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
  RooArgList* part1IntList = _part1ListMgr.getNormListByIndex(code-1) ;
  RooArgList* part2IntList = _part2ListMgr.getNormListByIndex(code-1) ;
  RooArgList* cod1List = _cod1ListMgr.getNormListByIndex(code-1) ;
  RooArgList* cod2List = _cod2ListMgr.getNormListByIndex(code-1) ;
  return calculate(part1IntList,cod1List)/calculate(part2IntList,cod2List) ;
}



Bool_t RooProdPdf::checkDependents(const RooArgSet* nset) const 
{
  // Check that none of the PDFs have overlapping dependents


  // Reduce nset to real-valued dependents only. Category dependent overlaps are allowed
  if (!nset) return kFALSE ;
  RooArgSet nsetReal ;
  TIterator* iter = nset->createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (dynamic_cast<RooAbsReal*>(arg)) {
      nsetReal.add(*arg) ;
    }
  }
  delete iter ;
  
  Bool_t ret(kFALSE) ;
  
  _pdfIter->Reset() ;
  RooAbsPdf* pdf, *pdf2 ;
  TIterator* iter2 = _pdfList.createIterator() ;
  while(pdf = (RooAbsPdf*)_pdfIter->Next()) {
    *iter2 = *_pdfIter ;
    while(pdf2 = (RooAbsPdf*)iter2->Next()) {
      if (pdf->dependentOverlaps(&nsetReal,*pdf2)) {

	cout << "RooProdPdf::checkDependents(" << GetName() << "): ERROR: PDFs " << pdf->GetName() 
	     << " and " << pdf2->GetName() << " have one or more real-valued dependents in common" << endl ;
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



Bool_t RooProdPdf::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange) 
{
  Bool_t ret(kFALSE) ;  

  Int_t i ;
  for (i=0 ; i<_part1ListMgr.cacheSize() ; i++) {
    RooArgList* plist = _part1ListMgr.getNormListByIndex(i) ;
    TIterator* iter = plist->createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
    }
    delete iter ;
  }

  for (i=0 ; i<_part2ListMgr.cacheSize() ; i++) {
    RooArgList* plist = _part2ListMgr.getNormListByIndex(i) ;
    TIterator* iter = plist->createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
      ret |= arg->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange) ;
    }
    delete iter ;
  }

  return ret ;
}
