/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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
