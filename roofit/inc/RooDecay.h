/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDecay.rdl,v 1.1 2001/06/08 05:52:38 verkerke Exp $
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
  virtual TObject* clone() const { return new RooDecay(*this) ; }
  virtual ~RooDecay();

  virtual Double_t coefficient(Int_t basisIndex) const ;
  
protected:
  
  Int_t _basisIdxPlus ;
  Int_t _basisIdxMinus ;

  ClassDef(RooDecay,1) // Abstract Resolution Model
};

#endif
