/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File:
 * Authors:
 *   PB, Paul Bloom, CU Boulder, bloom@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_BCPGEN_DECAY
#define ROO_BCPGEN_DECAY

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooCategoryProxy.hh"

class RooBCPGenDecay : public RooConvolutedPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooBCPGenDecay() { }
  RooBCPGenDecay(const char *name, const char *title, 
		 RooRealVar& t, RooAbsCategory& tag,
		 RooAbsReal& tau, RooAbsReal& dm,
		 RooAbsReal& avgMistag, 
		 RooAbsReal& a, RooAbsReal& b,
		 RooAbsReal& delMistag,
		 const RooResolutionModel& model, DecayType type=DoubleSided) ;

  RooBCPGenDecay(const RooBCPGenDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooBCPGenDecay(*this,newname) ; }
  virtual ~RooBCPGenDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  virtual Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars) const;
  void initGenerator(Int_t code) ;
  void generateEvent(Int_t code) ;
  
protected:

  RooRealProxy _C ;
  RooRealProxy _S ;
  RooRealProxy _avgMistag ;
  RooRealProxy _delMistag ;
  RooRealProxy _t ;
  RooRealProxy _tau ;
  RooRealProxy _dm ;
  RooCategoryProxy _tag ;
  Double_t _genB0Frac ;
  
  DecayType _type ;
  Int_t _basisExp ;
  Int_t _basisSin ;
  Int_t _basisCos ;

  ClassDef(RooBCPGenDecay,1) 
};

#endif
