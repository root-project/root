/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooResolutionModel.rdl,v 1.1 2001/06/08 05:51:06 verkerke Exp $
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
  RooResolutionModel* convolution(RooFormulaVar* basis) const ;
  const RooFormulaVar& basis() const ;
  const RooRealVar& convVar() const ;

  inline Bool_t isBasisSupported(const char* name) const { return basisCode(name)?kTRUE:kFALSE ; }
  virtual Int_t basisCode(const char* name) const = 0 ;

protected:

  RooRealProxy x ; // Dependent/convolution variable

  virtual Bool_t redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll) ;
  virtual void changeBasis(RooFormulaVar* basis) ;

  Int_t _basisCode ;
  RooFormulaVar* _basis ;

  ClassDef(RooResolutionModel,1) // Abstract Resolution Model
};

#endif
