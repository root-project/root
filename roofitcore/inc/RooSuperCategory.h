/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSuperCategory.rdl,v 1.4 2001/05/14 22:54:22 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_SUPER_CATEGORY
#define ROO_SUPER_CATEGORY

#include "TObjArray.h"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooCatType.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooSetProxy.hh"
 

class RooSuperCategory : public RooAbsCategoryLValue {
public:
  // Constructors etc.
  inline RooSuperCategory() { }
  RooSuperCategory(const char *name, const char *title, RooArgSet& inputCatList);
  RooSuperCategory(const RooSuperCategory& other, const char *name=0) ;
  virtual TObject* clone() const { return new RooSuperCategory(*this); }
  virtual ~RooSuperCategory();

  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  TIterator* MakeIterator() const ;
  const RooArgSet& inputCatList() const { return _catSet ; }

protected:

  Bool_t setType(const RooCatType* type, Bool_t prinError=kTRUE) ;
  void updateIndexList() ;
  TString currentLabel() const ;

  RooSetProxy _catSet ;
  
  virtual RooCatType evaluate() const ; 

  ClassDef(RooSuperCategory,1) // Derived index variable represening the maximal permutation of a list of indeces
};

#endif
