/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, kirkby@hep.uci.edu
 * History:
 *   25-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

#ifndef ROO_CHI2_VAR
#define ROO_CHI2_VAR

#include "RooFitCore/RooAbsOptGoodnessOfFit.hh"
class RooDataHist ;

class RooChi2Var : public RooAbsOptGoodnessOfFit {
public:

  // Constructors, assignment etc
  RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
	    Bool_t extended=kFALSE, Int_t nCPU=1) ;
  RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
	    const RooArgSet& projDeps, Bool_t extended=kFALSE, Int_t nCPU=1) ;
  RooChi2Var(const RooChi2Var& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooChi2Var(*this,newname); }

  virtual RooAbsGoodnessOfFit* create(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
				      const RooArgSet& projDeps, Int_t nCPU=1) {
    return new RooChi2Var(name,title,pdf,(RooDataHist&)data,projDeps,_extended,nCPU) ;
  }
  
  virtual ~RooChi2Var();

  virtual Double_t defaultErrorLevel() const { return 1.0 ; }

protected:

  Bool_t _extended ;
  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent) const ;
  
  ClassDef(RooChi2Var,1) // Abstract real-valued variable
};

#endif
