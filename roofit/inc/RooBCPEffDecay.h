/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBCPEffDecay.rdl,v 1.5 2001/12/13 22:09:35 schieti Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_BCP_EFF_DECAY
#define ROO_BCP_EFF_DECAY

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooCategoryProxy.hh"

class RooBCPEffDecay : public RooConvolutedPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooBCPEffDecay() { }
  RooBCPEffDecay(const char *name, const char *title, 
		 RooRealVar& t, RooAbsCategory& tag,
		 RooAbsReal& tau, RooAbsReal& dm,
		 RooAbsReal& avgMistag, RooAbsReal& CPeigenval,
		 RooAbsReal& a, RooAbsReal& b,
		 RooAbsReal& effRatio, RooAbsReal& delMistag,
		 const RooResolutionModel& model, DecayType type=DoubleSided) ;

  RooBCPEffDecay(const RooBCPEffDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooBCPEffDecay(*this,newname) ; }
  virtual ~RooBCPEffDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  virtual Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void initGenerator(Int_t code) ;
  void generateEvent(Int_t code) ;
  
protected:

  RooRealProxy _absLambda ;
  RooRealProxy _argLambda ;
  RooRealProxy _effRatio ;
  RooRealProxy _CPeigenval ;
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

  ClassDef(RooBCPEffDecay,1) // B Mixing decay PDF
};

#endif
