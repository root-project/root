/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, University of California Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   05-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_UNBLIND_OFFSET
#define ROO_UNBLIND_OFFSET

#include "RooFitCore/RooAbsHiddenReal.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitModels/RooBlindTools.hh"

class RooUnblindOffset : public RooAbsHiddenReal {
public:
  // Constructors, assignment etc
  RooUnblindOffset() ;
  RooUnblindOffset(const char *name, const char *title, 
		      const char *blindString, Double_t scale, RooAbsReal& blindValue);
  RooUnblindOffset(const RooUnblindOffset& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooUnblindOffset(*this,newname); }  
  virtual ~RooUnblindOffset();

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  RooRealProxy _value ;
  RooBlindTools _blindEngine ;

  ClassDef(RooUnblindOffset,1) // Offset unblinding transformation
};

#endif
