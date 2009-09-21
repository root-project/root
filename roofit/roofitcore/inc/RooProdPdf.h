/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProdPdf.h,v 1.44 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_PROD_PDF
#define ROO_PROD_PDF

#include "Riosfwd.h"
#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooLinkedList.h"
#include "RooAICRegistry.h"
#include "RooCacheManager.h"
#include "RooObjCacheManager.h"
#include "RooCmdArg.h"

typedef RooArgList* pRooArgList ;
typedef RooLinkedList* pRooLinkedList ;

class RooProdPdf : public RooAbsPdf {
public:
  RooProdPdf() ;
  RooProdPdf(const char *name, const char *title, Double_t cutOff=0);
  RooProdPdf(const char *name, const char *title,
	    RooAbsPdf& pdf1, RooAbsPdf& pdf2, Double_t cutOff=0) ;
  RooProdPdf(const char* name, const char* title, const RooArgList& pdfList, Double_t cutOff=0) ;
  RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet, const RooLinkedList& cmdArgList) ;

  RooProdPdf(const char* name, const char* title, const RooArgSet& fullPdfSet,
   	     const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),
             const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
             const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
             const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) ;

  RooProdPdf(const char* name, const char* title, 
             const RooCmdArg& arg1,             const RooCmdArg& arg2=RooCmdArg(),
             const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
             const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
             const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) ;

  RooProdPdf(const RooProdPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooProdPdf(*this,newname) ; }
  virtual ~RooProdPdf() ;

  virtual Double_t getVal(const RooArgSet* set=0) const ;
  Double_t evaluate() const ;
  virtual Bool_t checkObservables(const RooArgSet* nset) const ;	

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const ; 
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  virtual ExtendMode extendMode() const ;
  virtual Double_t expectedEvents(const RooArgSet* nset) const ; 
  virtual Double_t expectedEvents(const RooArgSet& nset) const { return expectedEvents(&nset) ; }

  const RooArgList& pdfList() const { return _pdfList ; }

  virtual Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  virtual void initGenerator(Int_t code) ;
  virtual void generateEvent(Int_t code);  
  virtual Bool_t isDirectGenSafe(const RooAbsArg& arg) const ; 

  // Constraint management
  virtual RooArgSet* getConstraints(const RooArgSet& observables, const RooArgSet& constrainedParams, Bool_t stripDisconnected) const ;

  virtual std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const ; 

  void printMetaArgs(ostream& os) const ;


protected:

  virtual void getParametersHook(const RooArgSet* /*nset*/, RooArgSet* /*list*/, Bool_t stripDisconnected) const ;

  void initializeFromCmdArgList(const RooArgSet& fullPdfSet, const RooLinkedList& l) ;

  void factorizeProduct(const RooArgSet& normSet, const RooArgSet& intSet, 
                        RooLinkedList& termList,   RooLinkedList& normList, 
                        RooLinkedList& impDepList, RooLinkedList& crossDepList,
                        RooLinkedList& intList) const;
  const char* makeRGPPName(const char* pfx, const RooArgSet& term, const RooArgSet& iset, const RooArgSet& nset, const char* isetRangeName) const ;
  void groupProductTerms(RooLinkedList& groupedTerms, RooArgSet& outerIntDeps,
                         const RooLinkedList& terms, const RooLinkedList& norms, 
                         const RooLinkedList& imps, const RooLinkedList& ints, const RooLinkedList& cross) const ;
  
  Double_t calculate(const RooArgList* partIntList, const RooLinkedList* normSetList) const ;
	
	
  void getPartIntList(const RooArgSet* nset, const RooArgSet* iset, pRooArgList& partList, pRooLinkedList& nsetList, 
                      Int_t& code, const char* isetRangeName=0) const ;
  RooAbsReal* processProductTerm(const RooArgSet* nset, const RooArgSet* iset, const char* isetRangeName,
                                 const RooArgSet* term,const RooArgSet& termNSet, const RooArgSet& termISet, 
                                 Bool_t& isOwned, Bool_t forceWrap=kFALSE) const ;


  // The cache object
  class CacheElem : public RooAbsCacheElement {
  public:
    virtual ~CacheElem() ;
    // Payload
    RooArgList _partList ;
    RooArgList _ownedList ;
    RooLinkedList _normList ;    
    // Cache management functions
    virtual RooArgList containedArgs(Action) ;
    virtual void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) ;
  } ;
  mutable RooObjCacheManager _cacheMgr ; // The cache manager
 
  friend class RooProdGenContext ;
  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype=0, 
	                               const RooArgSet *auxProto=0, Bool_t verbose= kFALSE) const ;

  RooArgSet* findPdfNSet(RooAbsPdf& pdf) const ; 

  mutable RooAICRegistry _genCode ; //! Registry of composite direct generator codes

  mutable RooArgSet* _curNormSet ; //!
  Double_t _cutOff ;       //  Cutoff parameter for running product
  RooListProxy _pdfList ;  //  List of PDF components
  RooLinkedList _pdfNSetList ; // List of PDF component normalization sets
  TIterator* _pdfIter ;    //! Iterator of PDF list
  Int_t _extendedIndex ;   //  Index of extended PDF (if any) 

  void useDefaultGen(Bool_t flag=kTRUE) { _useDefaultGen = flag ; }
  Bool_t _useDefaultGen ; // Use default or distributed event generator
  
private:

  ClassDef(RooProdPdf,1) // PDF representing a product of PDFs
};


#endif
