/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   14-May-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_EFFICIENCY
#define ROO_EFFICIENCY

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooCategoryProxy.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "TString.h" 

class RooArgList ;


class RooEfficiency : public RooAbsPdf {
public:
  // Constructors, assignment etc
  inline RooEfficiency() { }
  RooEfficiency(const char *name, const char *title, const RooAbsReal& effFunc, const RooAbsCategory& cat, const char* sigCatName);
  RooEfficiency(const RooEfficiency& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooEfficiency(*this,newname); }
  virtual ~RooEfficiency();

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;
  RooCategoryProxy _cat ; // Accept/reject categort
  RooRealProxy _effFunc ; // Efficiency modeling function
  TString _sigCatName ;   // Name of accept state of accept/reject category

  ClassDef(RooEfficiency,1) // Generic PDF defined by string expression and list of variables
};

#endif
