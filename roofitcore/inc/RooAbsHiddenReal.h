/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   19-Nov-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_HIDDEN_REAL
#define ROO_ABS_HIDDEN_REAL

class RooArgSet ;
#include "RooFitCore/RooAbsReal.hh"

class RooAbsHiddenReal : public RooAbsReal {
public:
  // Constructors, assignment etc.
  inline RooAbsHiddenReal() { }
  RooAbsHiddenReal(const char *name, const char *title, const char *unit= "") ;
  RooAbsHiddenReal(const RooAbsHiddenReal& other, const char* name=0) ;
  virtual ~RooAbsHiddenReal();
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  
protected:

  // This is dubious from a C++ point of view, but it blocks the interactive user
  // from accidentally calling getVal() without explicit cast, which is the whole
  // point of this class
  virtual Double_t getVal(const RooArgSet* nset=0) const { return RooAbsReal::getVal(nset) ; }

  ClassDef(RooAbsHiddenReal,1) // Abstract hidden real-valued variable
};

#endif
