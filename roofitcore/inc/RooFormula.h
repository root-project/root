/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $$
 * Authors:
 *   WV, Wouter Verkerke, University of California Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_FORMULA
#define ROO_FORMULA

#include "Rtypes.h"
#include "TFormula.h"
#include "RooFitCore/RooValue.hh"
#include "RooFitCore/RooArgSet.hh"

class RooArgSet ;
typedef RooValue* pRV ;
const Int_t kMAXRV = 50;

class RooFormula : public TFormula {
public:
  RooFormula() ;
  RooFormula(const char* name, const char* formula, RooArgSet& varList);
  RooFormula(const RooFormula& other) ;
  virtual ~RooFormula();
	
  RooArgSet& actualDependents() ;
  inline Double_t Eval() { return EvalPar(0,0) ; }
  Bool_t changeDependents(RooArgSet& newDeps, Bool_t mustReplaceAll=kFALSE) ;
  void dump() ;

protected:

  Int_t DefinedVariable(TString &name) ;
  Double_t DefinedValue(Int_t code) ;

  RooArgSet* _origList ;
  pRV _useList[kMAXRV] ;
  Int_t _nUse ;
  ClassDef(RooFormula,1) 
};

#endif
