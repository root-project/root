/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdPdf.cc,v 1.62 2005/12/08 15:26:16 wverkerke Exp $
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
// RooProdPdf is an efficient implementation of a product of PDFs of the form 
//
//  PDF_1 * PDF_2 * ... * PDF_N
//
// PDFs may share observables. If that is the case any irreducable subset
// of PDFS that share observables will be normalized with explicit numeric
// integration as any built-in normalization will no longer be valid.
//
// Alternatively, products using conditional PDFs can be defined, e.g.
//
//    F(x|y) * G(y)
//
// meaning a pdf F(x) _given_ y and a PDF G(y). In this contruction F is only
// normalized w.r.t x and G is normalized w.r.t y. The product in this construction
// is properly normalized.
//
// If exactly one of the component PDFs supports extended likelihood fits, the
// product will also be usable in extended mode, returning the number of expected
// events from the extendable component PDF. The extendable component does not
// have to appear in any specific place in the list.
// 

#include "RooFit.h"

#include "TIterator.h"
#include "TIterator.h"
#include "RooProdPdf.h"
#include "RooRealProxy.h"
#include "RooProdGenContext.h"
#include "RooGenProdProj.h"
#include "RooProduct.h"
#include "RooNameReg.h"

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
  RooArgSet* nset1 = new RooArgSet("nset1") ;
  _pdfNSetList.Add(nset1) ;
  if (pdf1.canBeExtended()) {
    _extendedIndex = _pdfList.index(&pdf1) ;
  }

  _pdfList.add(pdf2) ;
  RooArgSet* nset2 = new RooArgSet("nset2") ;
  _pdfNSetList.Add(nset2) ;

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
  RooAbsArg* arg ;
  Int_t numExtended(0) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (!pdf) {
      cout << "RooProdPdf::RooProdPdf(" << GetName() << ") list arg " 
	   << pdf->GetName() << " is not a PDF, ignored" << endl ;
      continue ;
    }
    _pdfList.add(*pdf) ;

    RooArgSet* nset = new RooArgSet("nset") ;
    _pdfNSetList.Add(nset) ;

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


RooProdPdf::RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet,
		       const RooCmdArg& arg1, const RooCmdArg& arg2,
		       const RooCmdArg& arg3, const RooCmdArg& arg4,
		       const RooCmdArg& arg5, const RooCmdArg& arg6,
		       const RooCmdArg& arg7, const RooCmdArg& arg8) :
  RooAbsPdf(name,title), 
  _partListMgr(10),
  _partOwnedListMgr(10),
  _genCode(10),
  _cutOff(0),
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _extendedIndex(-1),
  _useDefaultGen(kFALSE)
{
  // Constructor from named argument list
  //
  // fullPdf -- Set of 'regular' PDFS that are normalized over all their observables
  // ConditionalPdf(pdfSet,depSet) -- Add PDF to product with condition that it
  //                                  only be normalized over specified observables
  //                                  any remaining observables will be conditional
  //                                  observables
  //                               
  //
  // For example, given a PDF F(x,y) and G(y)
  //
  // RooProdPdf("P","P",G,Partial(F,x)) will construct a 2-dimensional PDF as follows:
  // 
  //   P(x,y) = G(y)/Int[y]G(y) * F(x,y)/Int[x]G(x,y)
  //
  // which is a well normalized and properly defined PDF, but different from the
  //  
  //  P'(x,y) = F(x,y)*G(y) / Int[x,y] F(x,y)*G(y)
  //
  // In the former case the y distribution of P is identical to that of G, while
  // F only is used to determine the correlation between X and Y. In the latter
  // case the Y distribution is defined by the product of F and G.
  //
  // This P(x,y) construction is analoguous to generating events from F(x,y) with
  // a prototype dataset sampled from G(y)
  
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  initializeFromCmdArgList(fullPdfSet,l) ;
}


RooProdPdf::RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet, const RooLinkedList& cmdArgList) :
  RooAbsPdf(name,title), 
  _partListMgr(10),
  _partOwnedListMgr(10),
  _genCode(10),
  _cutOff(0),
  _pdfList("_pdfList","List of PDFs",this),
  _pdfIter(_pdfList.createIterator()), 
  _extendedIndex(-1),
  _useDefaultGen(kFALSE)
{
//   cout << "RooProdPdf::ctor" << endl ;
//   cmdArgList.Print("v") ;
  initializeFromCmdArgList(fullPdfSet, cmdArgList) ;
}


RooProdPdf::RooProdPdf(const RooProdPdf& other, const char* name) :
  RooAbsPdf(other,name), 
  _partListMgr(other._partListMgr,kTRUE),
  _partOwnedListMgr(other._partOwnedListMgr,kTRUE),
  _genCode(other._genCode),
  _cutOff(other._cutOff),
  _pdfList("_pdfList",this,other._pdfList),
  _pdfIter(_pdfList.createIterator()), 
  _extendedIndex(other._extendedIndex),
  _useDefaultGen(other._useDefaultGen) 
{
  // Copy constructor

  // Clone contents of normalizarion set list
  TIterator* iter = other._pdfNSetList.MakeIterator() ;
  RooArgSet* nset ;
  while((nset=(RooArgSet*)iter->Next())) {
    _pdfNSetList.Add(nset->snapshot()) ;
  }
  delete iter ;

}



void RooProdPdf::initializeFromCmdArgList(const RooArgSet& fullPdfSet, const RooLinkedList& l)
{
  // Initialize RooProdPdf from a list of RooCmdArg configuration arguments

  Int_t numExtended(0) ;

  // Process set of full PDFS
  TIterator* siter = fullPdfSet.createIterator() ;
  RooAbsPdf* pdf ;
  while((pdf=(RooAbsPdf*)siter->Next())) {
    _pdfList.add(*pdf) ;
    RooArgSet* nset1 = new RooArgSet("nset1") ;
    _pdfNSetList.Add(nset1) ;       

    if (pdf->canBeExtended()) {
      _extendedIndex = _pdfList.index(pdf) ;
      numExtended++ ;
    }

  }
  delete siter ;

  // Process list of conditional PDFs
  TIterator* iter = l.MakeIterator() ;
  RooCmdArg* carg ;
  while((carg=(RooCmdArg*)iter->Next())) {
    if (!TString(carg->GetName()).CompareTo("Conditional")) {
   
      RooArgSet* pdfSet = (RooArgSet*) carg->getObject(0) ;
      RooArgSet* normSet = (RooArgSet*) carg->getObject(1) ;
      TIterator* siter = pdfSet->createIterator() ;
      RooAbsPdf* pdf ;
      while((pdf=(RooAbsPdf*)siter->Next())) {
	_pdfList.add(*pdf) ;
	_pdfNSetList.Add(normSet->snapshot()) ;       

	if (pdf->canBeExtended()) {
	  _extendedIndex = _pdfList.index(pdf) ;
	  numExtended++ ;
	}

      }
      delete siter ;


    } else if (TString(carg->GetName()).CompareTo("")) {
      cout << "Unknown arg: " << carg->GetName() << endl ;
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


RooProdPdf::~RooProdPdf()
{
  // Destructor
  Int_t i ;

  _pdfNSetList.Delete() ;
  for (i=0 ; i<10 ; i++) {
    _partNormListCache[i].Delete() ;
  }

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
  RooArgList *plist ;
  RooLinkedList *nlist ;
  getPartIntList(_curNormSet,0,plist,nlist,code) ;

  return calculate(plist,nlist) ;
}



Double_t RooProdPdf::calculate(const RooArgList* partIntList, const RooLinkedList* normSetList) const
{
  // Calculate running product of pdfs, skipping factorized components
  RooAbsReal* partInt ;
  RooArgSet* normSet ;
  Double_t value(1.0) ;
  Int_t n = partIntList->getSize() ;

  Int_t i ;
  for (i=0 ; i<n ; i++) {
    partInt = ((RooAbsReal*)partIntList->at(i)) ;
    normSet = ((RooArgSet*)normSetList->At(i)) ;    
    Double_t piVal = partInt->getVal(normSet->getSize()>0 ? normSet : 0) ;
    value *= piVal ;
//     if (_verboseEval<0) {
//       cout << "RPP:calc(" << GetName() << "): value *= " << piVal << " (" << partInt->GetName() << ") nset = " ; normSet->Print("1") ;
//     }
    if (value<_cutOff) {
      //cout << "RooProdPdf::calculate(" << GetName() << ") calculation cut off after " << partInt->GetName() << endl ; 
      break ;
    }
  }

  return value ;
}



void RooProdPdf::factorizeProduct(const RooArgSet& normSet, const RooArgSet& intSet,
				  RooLinkedList& termList, RooLinkedList& normList, 
				  RooLinkedList& impDepList, RooLinkedList& crossDepList,
				  RooLinkedList& intList) const 
{
  // Factorize product in irreducible terms for given choice of integration/normalization

//   cout << "RooProdPdf::factorizeProduct(" << GetName() << ")" << endl ;
//   cout << "   normSet  = " ; normSet.Print("1") ;
//   cout << "    intSet  = " ; intSet.Print("1") ;
 
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;

  // List of all term dependents: normalization and imported
  RooLinkedList depAllList ;
  RooLinkedList depIntNoNormList ;

  // Setup lists for factorization terms and their dependents
  RooArgSet* term(0) ;
  RooArgSet* termNormDeps(0) ;
  RooArgSet* termAllDeps(0) ;
  RooArgSet* termIntDeps(0) ;
  RooArgSet* termIntNoNormDeps(0) ;
  TIterator* lIter = termList.MakeIterator() ;
  TIterator* ldIter = normList.MakeIterator() ;
  TIterator* laIter = depAllList.MakeIterator() ;
  TIterator* nIter = _pdfNSetList.MakeIterator() ;
  RooArgSet* pdfNSet ;

  // Loop over the PDFs
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {    
    pdfNSet = (RooArgSet*) nIter->Next() ;
    lIter->Reset() ;
    ldIter->Reset() ;
    laIter->Reset() ;

    // Reduce pdfNSet to actual dependents
    pdfNSet = pdf->getObservables(*pdfNSet) ;
//     cout << "factorize: pdf = " << pdf->GetName() << " nset = " ; pdfNSet->Print("1") ;


    RooArgSet pdfNormDeps ; // Dependents to be normalized for the PDF
    RooArgSet pdfAllDeps ; // All dependents of this PDF ;

    // Make list of all dependents of this PDF
    RooArgSet* tmp = pdf->getObservables(normSet) ;
    pdfAllDeps.add(*tmp) ;
    delete tmp ;

    // Make list of normalization dependents for this PDF ;
    if (pdfNSet->getSize()>0) {
      RooArgSet* tmp = (RooArgSet*) pdfAllDeps.selectCommon(*pdfNSet) ;
      pdfNormDeps.add(*tmp) ;
      delete tmp ;
    } else {
      pdfNormDeps.add(pdfAllDeps) ;
    }

    RooArgSet* pdfIntSet = pdf->getObservables(intSet) ;

    RooArgSet pdfIntNoNormDeps(*pdfIntSet) ;
    pdfIntNoNormDeps.remove(pdfNormDeps,kTRUE,kTRUE) ;

//     cout << "factorize: all pdf dependents = " ; pdfAllDeps.Print("1") ;
//     cout << "factorize: normalized pdf dependents = " ; pdfNormDeps.Print("1") ;
//     cout << "factorize: integrated pdf dependents = " ; pdfIntSet->Print("1") ;
//     cout << "factorize: int/no-norm pdf dependents = " ; pdfIntNoNormDeps.Print("1") ;


    // Check if this PDF has dependents overlapping with one of the existing terms
    Bool_t done(kFALSE) ;
    while((term=(RooArgSet*)lIter->Next())) {      
      termNormDeps=(RooArgSet*)ldIter->Next() ;
      termAllDeps=(RooArgSet*)laIter->Next() ;

      // PDF should be added to existing term if 
      // 1) It has overlapping normalization dependents with any other PDF in existing term
      // 2) It has overlapping dependents of any class for which integration is requested

      Bool_t normOverlap = pdfNormDeps.overlaps(*termNormDeps)  ;
      // Bool_t intOverlap =  pdfIntSet->overlaps(*termAllDeps) ;

      //if (normOverlap || intOverlap) {
      if (normOverlap) {
// 	cout << "adding pdf " << pdf->GetName() << " to term " ; term->Print("1") ;
// 	cout << "   with termNormDeps " ; termNormDeps->Print("1") ;
	term->add(*pdf) ;
	termNormDeps->add(pdfNormDeps,kFALSE) ;
	termAllDeps->add(pdfAllDeps,kFALSE) ;
	termIntDeps->add(*pdfIntSet,kFALSE) ;
	termIntNoNormDeps->add(pdfIntNoNormDeps,kFALSE) ;
	done = kTRUE ;
      }
    }

    // If not, create a new term
    if (!done) {
//       cout << "creating new term" << endl ;
      term = new RooArgSet("term") ;
      termNormDeps = new RooArgSet("termNormDeps") ;
      termAllDeps = new RooArgSet("termAllDeps") ;
      termIntDeps = new RooArgSet("termIntDeps") ;
      termIntNoNormDeps = new RooArgSet("termIntNoNormDeps") ;

      term->add(*pdf) ;
      termNormDeps->add(pdfNormDeps,kFALSE) ;
      termAllDeps->add(pdfAllDeps,kFALSE) ;
      termIntDeps->add(*pdfIntSet,kFALSE) ;
      termIntNoNormDeps->add(pdfIntNoNormDeps,kFALSE) ;

      termList.Add(term) ;
      normList.Add(termNormDeps) ;
      depAllList.Add(termAllDeps) ;
      intList.Add(termIntDeps) ;
      depIntNoNormList.Add(termIntNoNormDeps) ;
    }

    // We own the reduced version of pdfNSet
    delete pdfNSet ;
    delete pdfIntSet ;
  }

  // Loop over list of terms again to determine 'imported' observables
  lIter->Reset() ;
  ldIter->Reset() ;
  laIter->Reset() ;
  TIterator* innIter = depIntNoNormList.MakeIterator() ;

  while((term=(RooArgSet*)lIter->Next())) {
    RooArgSet* normDeps = (RooArgSet*) ldIter->Next() ;
    RooArgSet* allDeps = (RooArgSet*) laIter->Next() ;
    RooArgSet* intNoNormDeps = (RooArgSet*) innIter->Next() ;

    // Make list of wholly imported dependents
    RooArgSet impDeps(*allDeps) ;
    impDeps.remove(*normDeps,kTRUE,kTRUE) ;
    impDepList.Add(impDeps.snapshot()) ;

    // Make list of cross dependents (term is self contained for these dependents, 
    // but components import dependents from other components)
    RooArgSet* crossDeps = (RooArgSet*) intNoNormDeps->selectCommon(*normDeps) ;
    crossDepList.Add(crossDeps->snapshot()) ;
    delete crossDeps ;
  }


//    lIter->Reset() ;
//    cout << "list of terms:" << endl ;
//    while((term=(RooArgSet*)lIter->Next())) {
//      term->Print("1") ;
//    }

  depAllList.Delete() ;
  depIntNoNormList.Delete() ;

  delete nIter ;
  delete lIter ;
  delete ldIter ;
  delete laIter ;
  delete innIter ;

  return ;
}




void RooProdPdf::getPartIntList(const RooArgSet* nset, const RooArgSet* iset, 
				pRooArgList& partList, pRooLinkedList& nsetList, 
				Int_t& code, const char* isetRangeName) const 
{
  // Check if this configuration was created before
  Int_t sterileIdx(-1) ;
  RooArgList* partIntList = _partListMgr.getNormList(this,nset,iset,&sterileIdx,RooNameReg::ptr(isetRangeName)) ;
  if (partIntList) {
    code = _partListMgr.lastIndex() ;
    
    partList = partIntList ;
    nsetList = _partNormListCache+code ; 

    return ;
  }

  // Create containers for partial integral components to be generated
  partIntList = new RooArgList("partIntList") ;
  RooArgList* partIntOwnedList = new RooArgList("partIntList") ;
  RooLinkedList* partIntNormList = new RooLinkedList ;

  // Factorize the product in irreducible terms for this nset
  RooLinkedList terms, norms, imp, ints, cross ;
  factorizeProduct(nset?(*nset):RooArgSet(),iset?(*iset):RooArgSet(),terms,norms,imp,cross,ints) ;

  RooArgSet *norm, *integ, *xdeps ;
  
  // Group irriducible terms that need to be (partially) integrated together
  RooLinkedList groupedList ; 
  RooArgSet outerIntDeps ;
  groupProductTerms(groupedList,outerIntDeps,terms,norms,imp,ints,cross) ;
  TIterator* gIter = groupedList.MakeIterator() ;
  RooLinkedList* group ;
  
  while((group=(RooLinkedList*)gIter->Next())) {

//     group->Print("1") ;
    
    if (group->GetSize()==1) {
      RooArgSet* term = (RooArgSet*) group->At(0) ;

      Int_t termIdx = terms.IndexOf(term) ;
      norm=(RooArgSet*) norms.At(termIdx) ;
      integ=(RooArgSet*) ints.At(termIdx) ;
      xdeps=(RooArgSet*)cross.At(termIdx) ;

      RooArgSet termNSet, termISet, termXSet ;
      
      // Take list of normalization, integrated dependents from factorization algorithm
      termISet.add(*integ) ;
      termNSet.add(*norm) ;
      
      // Cross-imported integrated dependents
      termXSet.add(*xdeps) ;
      
      // Add prefab term to partIntList. 
      Bool_t isOwned ;
      RooAbsReal* func = processProductTerm(nset,iset,isetRangeName,term,termNSet,termISet,isOwned) ;
      if (func) {
	partIntList->add(*func) ;
	if (isOwned) partIntOwnedList->addOwned(*func) ;
	partIntNormList->Add(norm->snapshot()) ;
      }
    } else {

//       cout << "prod: composite group encountered!" << endl ;

      RooArgSet compTermSet, compTermNorm ;
      TIterator* tIter = group->MakeIterator() ;
      RooArgSet* term ;
      while((term=(RooArgSet*)tIter->Next())) {
	
	Int_t termIdx = terms.IndexOf(term) ;
	norm=(RooArgSet*) norms.At(termIdx) ;
	integ=(RooArgSet*) ints.At(termIdx) ;
	xdeps=(RooArgSet*)cross.At(termIdx) ;
	
	RooArgSet termNSet, termISet, termXSet ;
	termISet.add(*integ) ;
	termNSet.add(*norm) ;
	termXSet.add(*xdeps) ;

	// Remove outer integration dependents from termISet
	termISet.remove(outerIntDeps,kTRUE,kTRUE) ;
// 	cout << "termISet = " ; termISet.Print("1") ;

	Bool_t isOwned ;
	RooAbsReal* func = processProductTerm(nset,iset,isetRangeName,term,termNSet,termISet,isOwned,kTRUE) ;
// 	cout << "created composite term component " << func->GetName() << endl ;
	if (func) {
	  compTermSet.add(*func) ;
	  if (isOwned) partIntOwnedList->addOwned(*func) ;
	  compTermNorm.add(*norm,kFALSE) ;
	}      
      }

//       cout << "constructing special composite product" << endl ;
//       cout << "compTermSet = " ; compTermSet.Print("1") ;

      // WVE this doesn't work (yet) for some reason
      //const char* name = makeRGPPName("SPECPROJ_",compTermSet,outerIntDeps,RooArgSet()) ;
      //RooAbsReal* func = new RooGenProdProj(name,name,compTermSet,outerIntDeps,RooArgSet()) ;
      //partIntList->add(*func) ;
      //partIntOwnedList->addOwned(*func) ;

      // WVE but this does, so we'll keep it for the moment
      const char* prodname = makeRGPPName("SPECPROD_",compTermSet,outerIntDeps,RooArgSet(),isetRangeName) ;
      RooProduct* prodtmp = new RooProduct(prodname,prodname,compTermSet) ;
      partIntOwnedList->addOwned(*prodtmp) ;

      const char* intname = makeRGPPName("SPECINT_",compTermSet,outerIntDeps,RooArgSet(),isetRangeName) ;
      RooRealIntegral* inttmp = new RooRealIntegral(intname,intname,*prodtmp,outerIntDeps,0,0,isetRangeName) ;

      partIntOwnedList->addOwned(*inttmp) ;
      partIntList->add(*inttmp);

      partIntNormList->Add(compTermNorm.snapshot()) ;

      delete tIter ;      
    }
  }
  delete gIter ;

  // Store the partial integral list and return the assigned code ;
  code = _partListMgr.setNormList(this,nset,iset,partIntList,RooNameReg::ptr(isetRangeName)) ;
  _partOwnedListMgr.setNormList(this,nset,iset,partIntOwnedList,RooNameReg::ptr(isetRangeName)) ;

  // Store the normalization set list in the cache using the index code of _partListMgr
  _partNormListCache[code].Delete() ;
  // We own content of 'norms' and transfer ownership to the cache
  _partNormListCache[code] = *partIntNormList ;


  // WVE PRINT SECTION
//       cout << "RooProdPdf::getPIL(" << GetName() << "," << this << ") creating new configuration with code " << code << endl
//            << "    nset = " ; if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
//       cout << "    iset" << (isetRangeName?(Form("[%s]",isetRangeName)):"") << " = " ; if (iset) iset->Print("1") ; else cout << "<none>" << endl ;
//       cout << "    Partial Integral List:" ;
//       partIntList->Print("1") ;
//       cout << "    Partial Owned Integral List:" ;
//       partIntOwnedList->Print("1") ;
//       cout << "    Normalization list: " << endl ; partIntNormList->Print("1") ; 
//       cout << endl  ;
  
  partList =  partIntList ;
  nsetList = _partNormListCache+code ;


  // We own contents of all lists filled by factorizeProduct() 
  terms.Delete() ;
  ints.Delete() ;
  imp.Delete() ;
  norms.Delete() ;
  cross.Delete() ;
}


void RooProdPdf::groupProductTerms(RooLinkedList& groupedTerms, RooArgSet& outerIntDeps, 
				   const RooLinkedList& terms, const RooLinkedList& norms, 
				   const RooLinkedList& imps, const RooLinkedList& ints, const RooLinkedList& /*cross*/) const
{
  // Start out with each term in its own group
  TIterator* tIter = terms.MakeIterator() ;
  RooArgSet* term ;
  while((term=(RooArgSet*)tIter->Next())) {
    RooLinkedList* group = new RooLinkedList ;
    group->Add(term) ;
    groupedTerms.Add(group) ;
  }
  delete tIter ;

  // Make list of imported dependents that occur in any term
  RooArgSet allImpDeps ;
  TIterator* iIter = imps.MakeIterator() ;
  RooArgSet *impDeps ;
  while((impDeps=(RooArgSet*)iIter->Next())) {
    allImpDeps.add(*impDeps,kFALSE) ;
  }
  delete iIter ;

  // Make list of integrated dependents that occur in any term
  RooArgSet allIntDeps ;
  iIter = ints.MakeIterator() ;
  RooArgSet *intDeps ;
  while((intDeps=(RooArgSet*)iIter->Next())) {
    allIntDeps.add(*intDeps,kFALSE) ;
  }
  delete iIter ;
  
//   cout << "Complete lists of imported dependens" ; allImpDeps.Print("1") ;
//   cout << "Complete lists of integrated dependens" ; allIntDeps.Print("1") ;
  
  RooArgSet* tmp = (RooArgSet*) allIntDeps.selectCommon(allImpDeps) ;
  outerIntDeps.removeAll() ;
  outerIntDeps.add(*tmp) ;
  delete tmp ;
//   cout << "Integrated Dependents that need 'outer' integration: " ; outerIntDeps.Print("1") ;

  // Now iteratively merge groups that should be (partially) integrated together
  TIterator* oidIter = outerIntDeps.createIterator() ;
  TIterator* glIter = groupedTerms.MakeIterator() ;
  RooAbsArg* outerIntDep ;
  while ((outerIntDep =(RooAbsArg*)oidIter->Next())) {
    
//     cout << "merge for outerIntDep " << outerIntDep->GetName() << endl ;
    
    // Collect groups that feature this dependent
    RooLinkedList* newGroup = 0 ;

    // Loop over groups
    RooLinkedList* group ;
    glIter->Reset() ;    
    Bool_t needMerge = kFALSE ;
    while((group=(RooLinkedList*)glIter->Next())) {

//       cout << "considering the following group:" << endl ;
//       group->Print("1") ;
//       cout << "----" << endl ;

      // See if any term in this group depends in any ay on outerDepInt
      RooArgSet* term ;
      TIterator* tIter = group->MakeIterator() ;
      while((term=(RooArgSet*)tIter->Next())) {

	Int_t termIdx = terms.IndexOf(term) ;
	RooArgSet* termNormDeps = (RooArgSet*) norms.At(termIdx) ;
	RooArgSet* termIntDeps = (RooArgSet*) ints.At(termIdx) ;
	RooArgSet* termImpDeps = (RooArgSet*) imps.At(termIdx) ;

	if (termNormDeps->contains(*outerIntDep) || 
	    termIntDeps->contains(*outerIntDep) || 
	    termImpDeps->contains(*outerIntDep)) {
	  needMerge = kTRUE ;
	}
	
      }

      if (needMerge) {
// 	cout << "group needs to be merged" << endl ;
	// Create composite group if not yet existing
	if (newGroup==0) {
	  newGroup = new RooLinkedList ;
	}
	
	// Add terms of this group to new term      
	tIter->Reset() ;
	while((term=(RooArgSet*)tIter->Next())) {
// 	  cout << "transferring group term to new merged group" << endl ;
	  newGroup->Add(term) ;	  
	}

	// Remove this group from list and delete it (but not its contents)
// 	cout << "removing group from groupList" << endl ;
	groupedTerms.Remove(group) ;
	delete group ;
      }
      delete tIter ;
    }
    // If a new group has been created to merge terms dependent on current outerIntDep, add it to group list
    if (newGroup) {
//       cout << "adding merged group to group list" << endl ;
      groupedTerms.Add(newGroup) ;
//       newGroup->Print("1") ;
//       cout << "..." << endl ;
    }

  }

  delete glIter ;
  delete oidIter ;
}



RooAbsReal* RooProdPdf::processProductTerm(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName,
				           const RooArgSet* term,const RooArgSet& termNSet, const RooArgSet& termISet,
				           Bool_t& isOwned, Bool_t forceWrap) const
{
  // CASE I: factorizing term: term is integrated over all normalizing observables
  // -----------------------------------------------------------------------------
  // Check if all observbales of this term are integrated. If so the term cancels
  if (termNSet.getSize()>0 && termNSet.getSize()==termISet.getSize() && isetRangeName==0) {

    // Term factorizes    
    return 0 ;
  }
  
  // CASE II: Dropped terms: if term is entirely unnormalized, it should be dropped
  // ------------------------------------------------------------------------------
  if (nset && termNSet.getSize()==0) {
    
    // Drop terms that are not asked to be normalized  
    return 0 ;
  }
  
  if (iset && termISet.getSize()>0) {
    if (term->getSize()==1) {
      
      // CASE IIIa: Normalized and partially integrated single PDF term
      //---------------------------------------------------------------
      
      TIterator* pIter = term->createIterator() ;
      RooAbsPdf* pdf = (RooAbsPdf*) pIter->Next() ;
      delete pIter ;
      
      RooAbsReal* partInt = pdf->createIntegral(termISet,termNSet,isetRangeName) ;
      partInt->setOperMode(operMode()) ;

      isOwned=kTRUE ;
      return partInt ;
      
    } else {
      
      // CASE IIIb: Normalized and partially integrated composite PDF term
      //---------------------------------------------------------------
      
      // Use auxiliary class RooGenProdProj to calculate this term
      const char* name = makeRGPPName("GENPROJ_",*term,termISet,termNSet,isetRangeName) ;
      RooAbsReal* partInt = new RooGenProdProj(name,name,*term,termISet,termNSet,isetRangeName) ;
      partInt->setOperMode(operMode()) ;

      isOwned=kTRUE ;
      return partInt ;
    }      
  }
  
  // CASE IVa: Normalized non-integrated composite PDF term
  // -------------------------------------------------------
  if (nset && nset->getSize()>0 && term->getSize()>1) {
    // Composite term needs normalized integration
    const char* name = makeRGPPName("GENPROJ_",*term,termISet,termNSet,isetRangeName) ;
    RooAbsReal* partInt = new RooGenProdProj(name,name,*term,termISet,termNSet,isetRangeName) ;
    partInt->setOperMode(operMode()) ;

    isOwned=kTRUE ;
    return partInt ;
  }
  
  // CASE IVb: Normalized, non-integrated single PDF term 
  // -----------------------------------------------------
  TIterator* pIter = term->createIterator() ;
  RooAbsPdf* pdf ;
  while((pdf=(RooAbsPdf*)pIter->Next())) {

    if (forceWrap) {

      // Construct representative name of normalization wrapper
      TString name(pdf->GetName()) ;
      name.Append("_NORM[") ;
      TIterator* nIter = termNSet.createIterator() ;
      RooAbsArg* arg ;
      Bool_t first(kTRUE) ;
      while((arg=(RooAbsArg*)nIter->Next())) {
	if (!first) {
	  name.Append(",") ;
	} else {
	  first=kFALSE ;
	}		   
	name.Append(arg->GetName()) ;
      }      
      name.Append("]") ;
      delete nIter ;

      
      RooAbsReal* partInt = new RooRealIntegral(name.Data(),name.Data(),*pdf,RooArgSet(),&termNSet) ;
      isOwned=kTRUE ;      
      return partInt ;

    } else {
      isOwned=kFALSE ;
      return pdf  ;
    }
  }
  delete pIter ;

  cout << "RooProdPdf::processProductTerm(" << GetName() << ") unidentified term!!!" << endl ;
  return 0;
}




const char* RooProdPdf::makeRGPPName(const char* pfx, const RooArgSet& term, const RooArgSet& iset, const RooArgSet& nset, const char* isetRangeName) const
{
  // Make an appropriate name for a RooGenProdProj object in getPartIntList() 

  static TString pname ;
  pname = pfx ;

  TIterator* pIter = term.createIterator() ;
  
  // Encode component names
  Bool_t first(kTRUE) ;
  RooAbsPdf* pdf ;
  while((pdf=(RooAbsPdf*)pIter->Next())) {
    if (first) {
      first = kFALSE ;
    } else {
      pname.Append("_X_") ;
    }
    pname.Append(pdf->GetName()) ;
  }
  delete pIter ;

  pname.Append(integralNameSuffix(iset,&nset,isetRangeName)) ;  

  return pname.Data() ;
}


Bool_t RooProdPdf::forceAnalyticalInt(const RooAbsArg& /*dep*/) const 
{
  return kTRUE ;
}


Int_t RooProdPdf::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, 
					  const RooArgSet* normSet, const char* rangeName) const 
{
  // Determine which part (if any) of given integral can be performed analytically.
  // If any analytical integration is possible, return integration scenario code.
  //
  // RooProdPdf implements two strategies in implementing analytical integrals
  //
  // First, PDF components whose entire set of dependents are requested to be integrated
  // can be dropped from the product, as they will integrate out to 1 by construction
  //:
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
  RooArgList *plist ;
  RooLinkedList *nlist ;
  getPartIntList(normSet,&allVars,plist,nlist,code,rangeName) ;
  
  return code+1 ;
}




Double_t RooProdPdf::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const 
{
  // Return analytical integral defined by given scenario code

  // No integration scenario
  if (code==0) {
    return getVal(normSet) ;
  }

  // WVE needs adaptation for rangename feature

  // Partial integration scenarios
  RooArgList* partIntList = _partListMgr.getNormListByIndex(code-1) ;  

  // If cache has been sterilized, revive this slot
  if (partIntList==0) {
    RooArgSet* vars = getParameters(RooArgSet()) ;
    RooArgSet* nset = _partListMgr.nameSet1ByIndex(code-1)->select(*vars) ;
    RooArgSet* iset = _partListMgr.nameSet2ByIndex(code-1)->select(*vars) ;

    Int_t code2(-1) ;
    RooLinkedList *nlist ;
    getPartIntList(nset,iset,partIntList,nlist,code2,rangeName) ;

    delete vars ;
    delete nset ;
    delete iset ;
  }
  Double_t val = calculate(partIntList,_partNormListCache+code-1) ;
  
  //cout << "RPP::aIWN(" << GetName() << ") value = " << val << endl ;
  return val ;
}



Bool_t RooProdPdf::checkObservables(const RooArgSet* /*nset*/) const 
{
  // Check that none of the PDFs have overlapping dependents
  return kFALSE ;
  
//   Bool_t ret(kFALSE) ;
  
//   _pdfIter->Reset() ;
//   RooAbsPdf* pdf, *pdf2 ;
//   TIterator* iter2 = _pdfList.createIterator() ;
//   while((pdf = (RooAbsPdf*)_pdfIter->Next())) {
//     *iter2 = *_pdfIter ;
//     while((pdf2 = (RooAbsPdf*)iter2->Next())) {
//       if (pdf->observableOverlaps(nset,*pdf2)) {
// 	cout << "RooProdPdf::checkObservables(" << GetName() << "): ERROR: PDFs " << pdf->GetName() 
// 	     << " and " << pdf2->GetName() << " have one or more dependents in common" << endl ;
// 	ret = kTRUE ;
//       }    
//     }
//   }
//   delete iter2 ;
//   return ret ;
}




RooAbsPdf::ExtendMode RooProdPdf::extendMode() const
{
  return (_extendedIndex>=0) ? ((RooAbsPdf*)_pdfList.at(_extendedIndex))->extendMode() : CanNotBeExtended ;
}



Double_t RooProdPdf::expectedEvents(const RooArgSet* nset) const 
{
  assert(_extendedIndex>=0) ;
  return ((RooAbsPdf*)_pdfList.at(_extendedIndex))->expectedEvents(nset) ;
}




RooAbsGenContext* RooProdPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					 const RooArgSet* auxProto, Bool_t verbose) const 
{
  if (_useDefaultGen) return RooAbsPdf::genContext(vars,prototype,auxProto,verbose) ;
  return new RooProdGenContext(*this,vars,prototype,auxProto,verbose) ;
}



Int_t RooProdPdf::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (!_useDefaultGen) return 0 ;


  // Find the subset directVars that only depend on a single PDF in the product
  RooArgSet directSafe ;
  TIterator* dIter = directVars.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)dIter->Next())) {
    if (isDirectGenSafe(*arg)) directSafe.add(*arg) ;
  }
  delete dIter ;


  // Now find direct integrator for relevant components ;
  _pdfIter->Reset() ;
  RooAbsPdf* pdf ;
  Int_t code[64], n(0) ;
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {
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
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {
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
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {
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
    if (plist) {
      TIterator* iter = plist->createIterator() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)iter->Next())) {
	arg->setOperMode(_operMode) ;
      }
      delete iter ;
    }
  }
  return ;
}


void RooProdPdf::clearCache() 
{
//   cout << "RooProdPdf::clearCache(" << GetName() << ") sterilizing cache content" << endl ;
  Int_t i ;
  for (i=0 ; i<10 ; i++) {
    _partNormListCache[i].Delete() ;
  }
  _partListMgr.sterilize()  ;
  _partOwnedListMgr.sterilize()  ; 
}



Bool_t RooProdPdf::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, 
				       Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
  // Throw away cache, as figuring out redirections on the cache is an unsolvable problem. 
  clearCache() ;
  return kFALSE ;
}

void RooProdPdf::printCompactTreeHook(ostream& os, const char* indent) 
{
  Int_t i ;
  os << indent << "RooProdPdf begin partial integral cache" << endl ;

  for (i=0 ; i<_partListMgr.cacheSize() ; i++) {
    RooArgList* plist = _partListMgr.getNormListByIndex(i) ;    
    if (plist) {
      TIterator* iter = plist->createIterator() ;
      RooAbsArg* arg ;
      TString indent2(indent) ;
      indent2 += Form("[%d] ",i) ;
      while((arg=(RooAbsArg*)iter->Next())) {      
	arg->printCompactTree(os,indent2) ;
      }
      delete iter ;
    }
  }

  os << indent << "RooProdPdf end partial integral cache" << endl ;
}



Bool_t RooProdPdf::isDirectGenSafe(const RooAbsArg& arg) const 
{
  // Only override base class behaviour if default generator method is enabled
  if (!_useDefaultGen) return RooAbsPdf::isDirectGenSafe(arg) ;

  // Argument may appear in only one PDF component
  _pdfIter->Reset() ;
  RooAbsPdf* pdf, *thePdf(0) ;  
  while((pdf=(RooAbsPdf*)_pdfIter->Next())) {

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



RooArgSet* RooProdPdf::findPdfNSet(RooAbsPdf& pdf) const 
{
  // Look up user specified normalization set for given input PDF component

  Int_t idx = _pdfList.index(&pdf) ;
  if (idx<0) return 0 ;
  return (RooArgSet*) _pdfNSetList.At(idx) ;
}
