/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooResolutionModel.rdl,v 1.2 2001/06/09 05:08:48 verkerke Exp $
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

  Double_t getVal(const RooDataSet* dset=0) const ;
  virtual RooResolutionModel* convolution(RooFormulaVar* basis) const ;
  const RooFormulaVar& basis() const ;
  RooRealVar& convVar() const ;
  const RooRealVar& basisConvVar() const ;

  inline Bool_t isBasisSupported(const char* name) const { return basisCode(name)?kTRUE:kFALSE ; }
  virtual Int_t basisCode(const char* name) const = 0 ;

protected:

  friend class RooAddModel ;
  RooRealProxy x ; // Dependent/convolution variable

  virtual Bool_t redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll) ;
  virtual void changeBasis(RooFormulaVar* basis) ;
  Bool_t traceEvalHook(Double_t value) const ;

  Int_t _basisCode ;
  RooFormulaVar* _basis ;

  ClassDef(RooResolutionModel,1) // Abstract Resolution Model
};

#endif
