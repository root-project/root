/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTruthModel.rdl,v 1.10 2002/03/25 22:09:55 verkerke Exp $
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

  enum RooTruthBasis { noBasis=0, expBasisMinus= 1, expBasisSum= 2, expBasisPlus= 3,
                                  sinBasisMinus=11, sinBasisSum=12, sinBasisPlus=13,
                                  cosBasisMinus=21, cosBasisSum=22, cosBasisPlus=23,
		                                                    linBasisPlus=33,
		                                                   quadBasisPlus=43,
                       genericBasis=100 } ;

  enum BasisType { none=0, expBasis=1, sinBasis=2, cosBasis=3,
                   linBasis=4, quadBasis=5 } ;
  enum BasisSign { Both=0, Plus=+1, Minus=-1 } ;

  // Constructors, assignment etc
  inline RooTruthModel() { }
  RooTruthModel(const char *name, const char *title, RooRealVar& x) ; 
  RooTruthModel(const RooTruthModel& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooTruthModel(*this,newname) ; }
  virtual ~RooTruthModel();
  
  virtual Int_t basisCode(const char* name) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:
  virtual Double_t evaluate() const ;
  virtual void changeBasis(RooFormulaVar* basis) ;

  ClassDef(RooTruthModel,1) // Abstract Resolution Model
};

#endif
