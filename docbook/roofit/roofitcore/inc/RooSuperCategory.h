/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSuperCategory.h,v 1.17 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SUPER_CATEGORY
#define ROO_SUPER_CATEGORY

class TObject ;
#include "RooAbsCategoryLValue.h"
#include "RooCatType.h"
#include "RooArgSet.h"
#include "RooSetProxy.h"
 

class RooSuperCategory : public RooAbsCategoryLValue {
public:
  // Constructors etc.
  inline RooSuperCategory() { _catIter = _catSet.createIterator() ; }
  RooSuperCategory(const char *name, const char *title, const RooArgSet& inputCatList);
  RooSuperCategory(const RooSuperCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooSuperCategory(*this,newname); }
  virtual ~RooSuperCategory();

  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) ;

  // Printing interface (human readable)
  virtual void printMultiline(ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent="") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  TIterator* MakeIterator() const ;
  const RooArgSet& inputCatList() const { return _catSet ; }

  virtual Bool_t inRange(const char* rangeName) const ;
  virtual Bool_t hasRange(const char* rangeName) const ;

protected:

  Bool_t setType(const RooCatType* type, Bool_t prinError=kTRUE) ;
  void updateIndexList() ;
  TString currentLabel() const ;

  RooSetProxy _catSet ; // Set of input category
  TIterator* _catIter ; //! Iterator over set of input categories
  
  virtual RooCatType evaluate() const ; 

  ClassDef(RooSuperCategory,1) // Lvalue product operator for catategory lvalues
};

#endif
