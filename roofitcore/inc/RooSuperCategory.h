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
#ifndef ROO_SUPER_CATEGORY
#define ROO_SUPER_CATEGORY

#include "TObjArray.h"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooCatType.hh"
#include "RooFitCore/RooArgSet.hh"
 

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

protected:

  void updateIndexList() ;
  TString currentLabel() const ;

  RooArgSet _catList ;
  RooSuperCategory& operator=(const RooSuperCategory& other) ;
  
  virtual RooCatType evaluate() const ; 

  ClassDef(RooSuperCategory,1) // a integer-valued category variable
};

#endif
