/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_MULTI_CATEGORY
#define ROO_MULTI_CATEGORY

#include "TObjArray.h"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooCatType.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooSetProxy.hh"
 

class RooMultiCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooMultiCategory() { }
  RooMultiCategory(const char *name, const char *title, const RooArgSet& inputCatList);
  RooMultiCategory(const RooMultiCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooMultiCategory(*this,newname); }
  virtual ~RooMultiCategory();

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  const RooArgSet& inputCatList() const { return _catSet ; }

protected:

  void updateIndexList() ;
  TString currentLabel() const ;

  RooSetProxy _catSet ; // Set of input category
  
  virtual RooCatType evaluate() const ; 

  ClassDef(RooMultiCategory,1) // Derived index variable represening the maximal permutation of a list of indeces
};

#endif
