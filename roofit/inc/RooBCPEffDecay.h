/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooBCPEffDecay.rdl,v 1.1 2001/06/26 18:13:00 verkerke Exp $
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
		 RooAbsReal& a, RooRealVar& b,
		 RooAbsReal& effRatio, RooRealVar& delMistag,
		 const RooResolutionModel& model, DecayType type=DoubleSided) ;

  RooBCPEffDecay(const RooBCPEffDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooBCPEffDecay(*this,newname) ; }
  virtual ~RooBCPEffDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  virtual Int_t getCoefAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t coefAnalyticalIntegral(Int_t coef, Int_t code) const ;
  
protected:

  RooRealProxy _absLambda ;
  RooRealProxy _argLambda ;
  RooRealProxy _effRatio ;
  RooRealProxy _CPeigenval ;
  RooRealProxy _avgMistag ;
  RooRealProxy _delMistag ;
  RooCategoryProxy _tag ;

  Int_t _basisExpPlus ;
  Int_t _basisExpMinus ;
  Int_t _basisSinPlus ;
  Int_t _basisSinMinus ;
  Int_t _basisCosPlus ;
  Int_t _basisCosMinus ;

  ClassDef(RooBCPEffDecay,1) // B Mixing decay PDF
};

#endif
