/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTruthModel.rdl,v 1.5 2001/08/23 01:21:48 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_TRUTH_MODEL
#define ROO_TRUTH_MODEL

#include "RooFitCore/RooResolutionModel.hh"

class RooTruthModel : public RooResolutionModel {
public:

  enum RooTruthBasis { noBasis=0, genericBasis=1,
		       expBasisPlus=2, expBasisMinus=3,
		       sinBasisPlus=4, sinBasisMinus=5,
		       cosBasisPlus=6, cosBasisMinus=7  } ;

  // Constructors, assignment etc
  inline RooTruthModel() { }
  RooTruthModel(const char *name, const char *title, RooRealVar& x) ; 
  RooTruthModel(const RooTruthModel& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooTruthModel(*this,newname) ; }
  virtual ~RooTruthModel();
  
  virtual Int_t basisCode(const char* name) const ;

protected:
  virtual Double_t evaluate() const ;
  virtual void changeBasis(RooFormulaVar* basis) ;

  ClassDef(RooTruthModel,1) // Abstract Resolution Model
};

#endif
