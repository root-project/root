/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooMappedCategory.rdl,v 1.12 2001/05/14 22:54:21 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_MAPPED_CATEGORY
#define ROO_MAPPED_CATEGORY

#include "TObjArray.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooCategoryProxy.hh"
#include "RooFitCore/RooCatType.hh"

class RooMappedCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooMappedCategory() { }
  RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defCatName="NotMapped");
  RooMappedCategory(const RooMappedCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooMappedCategory(*this,newname); }
  virtual ~RooMappedCategory();

  // Mapping function
  Bool_t map(const char* inKeyRegExp, const char* outKeyName) ; 

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:
  
  TObjArray _mapArray ;
  RooCatType* _defCat ;
  RooCategoryProxy _inputCat ;

  virtual RooCatType evaluate() const ; 

  ClassDef(RooMappedCategory,1) // Index varibiable, derived from another index using pattern-matching based mapping
};

#endif
