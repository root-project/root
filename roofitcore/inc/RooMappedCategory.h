/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMappedCategory.rdl,v 1.21 2005/06/20 15:44:55 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_MAPPED_CATEGORY
#define ROO_MAPPED_CATEGORY

#include "TObjArray.h"
#include "RooAbsCategory.h"
#include "RooCategoryProxy.h"
#include "RooCatType.h"

class RooMappedCategory : public RooAbsCategory {
public:
  // Constructors etc.
  enum CatIdx { NoCatIdx=-99999 } ;
  inline RooMappedCategory() { }
  RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defCatName="NotMapped", Int_t defCatIdx=NoCatIdx);
  RooMappedCategory(const RooMappedCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooMappedCategory(*this,newname); }
  virtual ~RooMappedCategory();

  // Mapping function
  Bool_t map(const char* inKeyRegExp, const char* outKeyName, Int_t outKeyNum=NoCatIdx) ; 

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt=Standard, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:
  
  TObjArray _mapArray ;         // Array of mapping rules
  RooCatType* _defCat ;         // Default (unmapped) output type
  RooCategoryProxy _inputCat ;  // Input category

  virtual RooCatType evaluate() const ; 

  ClassDef(RooMappedCategory,1) // Index varibiable, derived from another index using pattern-matching based mapping
};

#endif
