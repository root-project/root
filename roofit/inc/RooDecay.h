/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDecay.rdl,v 1.4 2001/10/31 07:21:21 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_DECAY
#define ROO_DECAY

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooDecay : public RooConvolutedPdf {
public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  // Constructors, assignment etc
  inline RooDecay() { }
  RooDecay(const char *name, const char *title, RooRealVar& t, 
	   RooAbsReal& tau, const RooResolutionModel& model, DecayType type) ;
  RooDecay(const RooDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooDecay(*this,newname) ; }
  virtual ~RooDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars) const;
  void generateEvent(Int_t code);
  
protected:
  
  RooRealProxy _t ;
  RooRealProxy _tau ;
  DecayType    _type ;
  Int_t        _basisExp ;

  ClassDef(RooDecay,1) // Abstract Resolution Model
};

#endif
