/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAddModel.rdl,v 1.9 2001/09/25 01:15:58 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   19-Jun-2001 WV Initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#ifndef ROO_ADD_MODEL
#define ROO_ADD_MODEL

#include "RooFitCore/RooResolutionModel.hh"
#include "RooFitCore/RooAICRegistry.hh"
#include "TList.h"

class RooAddModel : public RooResolutionModel {
public:
  RooAddModel(const char *name, const char *title, const RooArgList& modelList, const RooArgList& coefList);
  RooAddModel(const RooAddModel& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooAddModel(*this,newname) ; }
  virtual RooResolutionModel* convolution(RooFormulaVar* basis, RooAbsArg* owner) const ;
  virtual ~RooAddModel() ;

  virtual Double_t evaluate() const ;
  virtual Double_t getNorm(const RooArgSet* nset=0) const ;
  virtual Bool_t checkDependents(const RooArgSet* nset) const ;	
  virtual Int_t basisCode(const char* name) const ;

  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const ;
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const ;
  virtual Bool_t selfNormalized() const { return kTRUE ; }

  virtual void syncNormalization(const RooArgSet* nset) const ;
  virtual void normLeafServerList(RooArgSet& list) const ;

protected:

  mutable RooAICRegistry _codeReg ;  // Registry of component analytical integration codes

  virtual Double_t getNormSpecial(const RooArgSet* nset=0) const ;

  Bool_t _isCopy ;
  TList _modelProxyList ;
  TList _coefProxyList ;
  TIterator* _modelProxyIter ;
  TIterator* _coefProxyIter ;

private:

  ClassDef(RooAddModel,0) // Resolution model consisting of a sum of resolution models
};

#endif
