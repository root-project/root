/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddPdf.rdl,v 1.33 2002/09/30 00:57:28 verkerke Exp $
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
#ifndef ROO_ADD_PDF
#define ROO_ADD_PDF

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooAICRegistry.hh"
#include "RooFitCore/RooNormSetCache.hh"
#include "RooFitCore/RooNameSet.hh"
#include "RooFitCore/RooNormListManager.hh"

class RooAddPdf : public RooAbsPdf {
public:

  RooAddPdf(const char *name, const char *title);
  RooAddPdf(const char *name, const char *title,
	    RooAbsPdf& pdf1, RooAbsPdf& pdf2, RooAbsReal& coef1) ;
  RooAddPdf(const char *name, const char *title, const RooArgList& pdfList) ;
  RooAddPdf(const char *name, const char *title, const RooArgList& pdfList, const RooArgList& coefList) ;
  RooAddPdf(const RooAddPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddPdf(*this,newname) ; }
  virtual ~RooAddPdf() ;

  Double_t evaluate() const ;
  virtual Bool_t checkDependents(const RooArgSet* nset) const ;	

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  virtual ExtendMode extendMode() const { return (_haveLastCoef || _allExtendable) ? MustBeExtended : CanNotBeExtended; }
  virtual Double_t expectedEvents() const ;

  const RooArgList& pdfList() const { return _pdfList ; }
  const RooArgList& coefList() const { return _coefList ; }

  void fixCoefNormalization(const RooArgSet& refCoefNorm) ;

protected:

  virtual void selectNormalization(const RooArgSet* depSet=0, Bool_t force=kFALSE) ;

  mutable RooSetProxy _refCoefNorm ;

  Bool_t _projectCoefs ;
  void syncCoefProjList(const RooArgSet* nset, const RooArgSet* iset=0) const ;
  mutable RooNormListManager _projListMgr ;
  mutable RooArgList* _pdfProjList ;

  void syncSuppNormList(const RooArgSet* nset) const ;
  mutable RooNormListManager _suppListMgr ;
  mutable RooArgSet* _lastSupNormSet ;

  void updateCoefCache(const RooArgSet* nset) const ;
  mutable Double_t* _coefCache ;
  
  friend class RooAddGenContext ;
  virtual RooAbsGenContext* genContext(const RooArgSet &vars, 
				       const RooDataSet *prototype=0, Bool_t verbose= kFALSE) const ;


  virtual void operModeHook() ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, 
				     Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  mutable RooAICRegistry _codeReg ;  // Registry of component analytical integration codes

  RooListProxy _pdfList ;   //  List of component PDFs
  RooListProxy _coefList ;  //  List of coefficients
  mutable RooArgList* _snormList ;  //  List of supplemental normalization factors
  TIterator* _pdfIter ;     //! Iterator over PDF list
  TIterator* _coefIter ;    //! Iterator over coefficient list
  
  Bool_t _haveLastCoef ;   //  Flag indicating if last PDFs coefficient was supplied in the ctor
  Bool_t _allExtendable ;   //  Flag indicating if all PDF components are extendable

private:

  ClassDef(RooAddPdf,0) // PDF representing a sum of PDFs
};

#endif
