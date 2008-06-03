/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealIntegral.h,v 1.44 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_INTEGRAL
#define ROO_REAL_INTEGRAL

#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooListProxy.h"

class RooArgSet ;
class TH1F ;
class RooAbsCategory ;
class RooRealVar ;
class RooAbsIntegrator ;
class RooNumIntConfig ;

class RooRealIntegral : public RooAbsReal {
public:

  // Constructors, assignment etc
  RooRealIntegral() ;
  RooRealIntegral(const char *name, const char *title, const RooAbsReal& function, const RooArgSet& depList,
		  const RooArgSet* funcNormSet=0, const RooNumIntConfig* config=0, const char* rangeName=0) ;
  RooRealIntegral(const RooRealIntegral& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooRealIntegral(*this,newname); }
  virtual ~RooRealIntegral();

  Bool_t isValid() const { return _valid; }

  void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  const RooArgSet& numIntCatVars() const { return _sumList ; }
  const RooArgSet& numIntRealVars() const { return _intList ; }
  const RooArgSet& anaIntVars() const { return _anaList ; }

protected:

  mutable Bool_t _valid;

  enum IntOperMode { Hybrid, Analytic, PassThrough } ;
  //friend class RooAbsPdf ;

  Bool_t initNumIntegrator() const;
  void autoSelectDirtyMode() ;

  virtual Double_t sum() const ;
  virtual Double_t integrate() const ;
  virtual Double_t jacobianProduct() const ;

  // Evaluation and validation implementation
  Double_t evaluate() const ;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ;
  Bool_t servesExclusively(const RooAbsArg* server,const RooArgSet& exclLVBranches) const ;


  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, 
				     Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;

  // Function pointer and integrands list
  mutable RooSetProxy _sumList ;
  mutable RooSetProxy _intList ;
  mutable RooSetProxy _anaList ;
  mutable RooSetProxy _jacList ;
  mutable RooSetProxy _facList ;
  mutable RooArgSet   _facListOwned ;
  TIterator*          _facListIter ; //! do not persist
  TIterator*          _jacListIter ; //! do not persist
  RooRealProxy       _function ; // must after set proxies
  RooArgSet*      _funcNormSet ;

  mutable RooArgSet       _saveInt ; //! do not persist
  mutable RooArgSet       _saveSum ; //! do not persist 

  RooNumIntConfig* _iconfig ;

  mutable RooListProxy _sumCat ; //! do not persist  
  TIterator* _sumCatIter ; //!
  
  Int_t _mode ;
  IntOperMode _intOperMode ;   // integration operation mode

  mutable Bool_t _restartNumIntEngine ; //! do not persist
  mutable RooAbsIntegrator* _numIntEngine ;  //! do not persist
  mutable RooAbsFunc *_numIntegrand;         //! do not persist

  TNamed* _rangeName ; 

  virtual void operModeHook() ; // cache operation mode

  ClassDef(RooRealIntegral,1) // Real-valued variable representing an integral over a RooAbsReal object
};

#endif
