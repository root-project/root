/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooResolutionModel.rdl,v 1.6 2001/08/10 22:22:54 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_RESOLUTION_MODEL
#define ROO_RESOLUTION_MODEL

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"

class RooResolutionModel : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooResolutionModel() { }
  RooResolutionModel(const char *name, const char *title, RooRealVar& x) ; 
  RooResolutionModel(const RooResolutionModel& other, const char* name=0);
  virtual TObject* clone(const char* newname) const = 0 ;
  virtual ~RooResolutionModel();

  Double_t getVal(const RooArgSet* nset=0) const ;
  virtual RooResolutionModel* convolution(RooFormulaVar* basis, RooAbsArg* owner) const ;
  RooRealVar& convVar() const ;
  const RooRealVar& basisConvVar() const ;

  inline Bool_t isBasisSupported(const char* name) const { return basisCode(name)?kTRUE:kFALSE ; }
  virtual Int_t basisCode(const char* name) const = 0 ;

  virtual void normLeafServerList(RooArgSet& list) const ;

  inline const RooFormulaVar& basis() const { return _basis?*_basis:*_identity ; }

protected:

  static RooFormulaVar* _identity ;

  friend class RooAddModel ;
  RooRealProxy x ; // Dependent/convolution variable

  virtual Bool_t syncNormalizationPreHook(RooAbsReal* norm,const RooArgSet* nset) const { return kTRUE ; } ;

  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll) ;
  virtual void changeBasis(RooFormulaVar* basis) ;
  Bool_t traceEvalHook(Double_t value) const ;

  Int_t _basisCode ;
  RooFormulaVar* _basis ;
  Bool_t _ownBasis ;

  ClassDef(RooResolutionModel,1) // Abstract Resolution Model
};

#endif
