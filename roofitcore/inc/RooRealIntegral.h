/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealIntegral.rdl,v 1.32 2002/09/05 04:33:53 verkerke Exp $
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
#ifndef ROO_REAL_INTEGRAL
#define ROO_REAL_INTEGRAL

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooSetProxy.hh"

class RooArgSet ;
class TH1F ;
class RooAbsCategory ;
class RooRealVar ;
class RooAbsIntegrator ;
class RooIntegratorConfig ;

class RooRealIntegral : public RooAbsReal {
public:

  // Constructors, assignment etc
  inline RooRealIntegral() : _valid(kFALSE),_numIntEngine(0),_numIntegrand(0) { }
  RooRealIntegral(const char *name, const char *title, const RooAbsReal& function, const RooArgSet& depList,
		  const RooArgSet* funcNormSet=0, const RooIntegratorConfig* config=0) ;
  RooRealIntegral(const RooRealIntegral& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooRealIntegral(*this,newname); }
  virtual ~RooRealIntegral();

  Bool_t isValid() const { return _valid; }

  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

protected:

  mutable Bool_t _valid;

  enum OperMode { Hybrid, Analytic, PassThrough } ;
  //friend class RooAbsPdf ;

  Bool_t initNumIntegrator() const;

  virtual Double_t sum() const ;
  virtual Double_t integrate() const ;
  virtual Double_t jacobianProduct() const ;

  // Evaluation and validation implementation
  Double_t evaluate() const ;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ;
  Bool_t servesExclusively(const RooAbsArg* server,const RooArgSet& exclLVBranches) const ;


  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, 
				     Bool_t mustReplaceAll, Bool_t nameChange) ;

  // Function pointer and integrands list
  mutable RooSetProxy _sumList ;
  mutable RooSetProxy _intList ;
  mutable RooSetProxy _anaList ;
  mutable RooSetProxy _jacList ;
  mutable RooSetProxy _facList ;
  TIterator*          _facListIter ; //! do not persist
  TIterator*          _jacListIter ; //! do not persist
  RooRealProxy       _function ; // must after set proxies
  RooArgSet*      _funcNormSet ;

  RooIntegratorConfig* _iconfig ;
  
  void prepareACleanFunc() const ;
  void restoreACleanFunc() const ;
  mutable RooArgSet _funcACleanBranchList ;

  Int_t _mode ;
  OperMode _operMode ;

  mutable Bool_t _restartNumIntEngine ; //! do not persist
  mutable RooAbsIntegrator* _numIntEngine ;  //! do not persist
  mutable RooAbsFunc *_numIntegrand;         //! do not persist


  ClassDef(RooRealIntegral,1) // Real-valued variable representing an integral over a RooAbsReal object
};

#endif
