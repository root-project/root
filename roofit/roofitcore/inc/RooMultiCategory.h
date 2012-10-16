/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMultiCategory.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_MULTI_CATEGORY
#define ROO_MULTI_CATEGORY

class TObject ;
#include "RooAbsCategoryLValue.h"
#include "RooCatType.h"
#include "RooArgSet.h"
#include "RooSetProxy.h"
 

class RooMultiCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooMultiCategory() { }
  RooMultiCategory(const char *name, const char *title, const RooArgSet& inputCatList);
  RooMultiCategory(const RooMultiCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooMultiCategory(*this,newname); }
  virtual ~RooMultiCategory();

  // Printing interface (human readable)
  virtual void printMultiline(std::ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const ;

  const RooArgSet& inputCatList() const { return _catSet ; }

protected:

  void updateIndexList() ;
  TString currentLabel() const ;

  RooSetProxy _catSet ; // Set of input category
  
  virtual RooCatType evaluate() const ; 

  ClassDef(RooMultiCategory,1) // Product operator for categories
};

#endif
