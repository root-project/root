/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddModel.rdl,v 1.21 2005/02/25 14:22:53 wverkerke Exp $
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

#ifndef ROO_ADD_MODEL
#define ROO_ADD_MODEL

#include "RooResolutionModel.h"
#include "RooAICRegistry.h"
#include "TList.h"

class RooAddModel : public RooResolutionModel {
public:
  RooAddModel(const char *name, const char *title, const RooArgList& modelList, const RooArgList& coefList);
  RooAddModel(const RooAddModel& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddModel(*this,newname) ; }
  virtual RooResolutionModel* convolution(RooFormulaVar* basis, RooAbsArg* owner) const ;
  virtual ~RooAddModel() ;

  virtual Double_t evaluate() const ;
  virtual Bool_t checkObservables(const RooArgSet* nset) const ;	
  virtual Int_t basisCode(const char* name) const ;

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const ;
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  Double_t getNorm(const RooArgSet* nset=0) const ;
  virtual Bool_t syncNormalization(const RooArgSet* nset, Bool_t adjustProxies=kTRUE) const ;
  virtual void normLeafServerList(RooArgSet& list) const ;

  virtual Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  virtual void initGenerator(Int_t code) ;
  virtual void generateEvent(Int_t code);  

  virtual Bool_t isDirectGenSafe(const RooAbsArg& arg) const ; 

protected:


  mutable RooNormSetCache _nsetCache; // Normalization set cache
  mutable RooAICRegistry _codeReg ;  //! Registry of component analytical integration codes
  mutable RooAICRegistry _genReg ;   //! Registry of component generator codes
  Double_t*  _genThresh ;            //! Generator fraction thresholds
  const Int_t* _genSubCode ;         //! Subgenerator code mapping (owned by _genReg) ;

  Bool_t _isCopy ;              // Flag set if we own our components
  RooRealProxy _dummyProxy ;    // Dummy proxy to hold current normalization set
  TList _modelProxyList ;       // List of component resolution models
  TList _coefProxyList ;        // List of coefficients
  TIterator* _modelProxyIter ;  //! Iterator over list of models
  TIterator* _coefProxyIter ;   //! Iterator over list of coefficients

private:

  ClassDef(RooAddModel,0) // Resolution model consisting of a sum of resolution models
};

#endif
