/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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

class RooResolutionModel : public RooAbsPdf {
public:

  // Constructors, assignment etc
  inline RooResolutionModel() { }
  RooResolutionModel(const char *name, const char *title) ; 
  RooResolutionModel(const RooResolutionModel& other, const char* name=0);
  virtual ~RooResolutionModel();

  Double_t getVal(const RooDataSet* dset) const ;
  RooResolutionModel* convolution(RooAbsReal* basis) const ;
  const RooAbsReal& basis() const ;
  const RooRealVar& convVar() const ;

  virtual Bool_t isBasisSupported(const char* name) const = 0 ;

protected:

  virtual Bool_t redirectServersHook(const RooArgSet& newServerList, Bool_t mustReplaceAll) ;
  void changeBasis(RooAbsReal* basis) ;

private:

  RooAbsReal* _basis ;
  ClassDef(RooResolutionModel,1) // Abstract Resolution Model
};

#endif
