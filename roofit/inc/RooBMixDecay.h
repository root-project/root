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
#ifndef ROO_BMIX_DECAY
#define ROO_BMIX_DECAY

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooCategoryProxy.hh"

class RooBMixDecay : public RooConvolutedPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooBMixDecay() { }
  RooBMixDecay(const char *name, const char *title, 
	       RooRealVar& t, RooAbsCategory& tag,
	       RooAbsReal& tau, RooAbsReal& dm,
	       RooAbsReal& mistag, const RooResolutionModel& model, 
	       DecayType type=DoubleSided) ;

  RooBMixDecay(const RooBMixDecay& other, const char* name=0);
  virtual TObject* clone() const { return new RooBMixDecay(*this) ; }
  virtual ~RooBMixDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  virtual Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code) const ;
  
protected:

  RooRealProxy     _mistag ;
  RooCategoryProxy _tag ;
  Int_t _basisExpPlus ;
  Int_t _basisExpMinus ;
  Int_t _basisCosPlus ;
  Int_t _basisCosMinus ;

  ClassDef(RooBMixDecay,1) // B Mixing decay PDF
};

#endif
