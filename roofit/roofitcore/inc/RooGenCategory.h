/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGenCategory.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_GEN_CATEGORY
#define ROO_GEN_CATEGORY

#include "RooAbsCategory.h"
#include "RooSuperCategory.h"
#include "RooCategoryProxy.h"
#include "RooCatType.h"

#include "TString.h"
class TObject ;

class RooGenCategory : public RooAbsCategory {
public:
  typedef const char* (*RooGetCategoryFunc_t)(RooArgSet&);
  // Constructors etc.
  inline RooGenCategory() { 
    // Default constructor
    // coverity[UNINIT_CTOR]
  }
  RooGenCategory(const char *name, const char *title, RooGetCategoryFunc_t userFunc, RooArgSet& catList);
  RooGenCategory(const RooGenCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooGenCategory(*this,newname); }
  virtual ~RooGenCategory();

  // Printing interface (human readable)
  virtual void printMultiline(std::ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const ;

protected:

  void initialize() ;
  TString evalUserFunc(RooArgSet *vars) ;
  void updateIndexList() ;
  
  RooSuperCategory _superCat ;      //  Super category of input categories
  RooCategoryProxy _superCatProxy ; // Proxy for super category
  Int_t *_map ;                     //! Super-index to generic-index map

  RooGetCategoryFunc_t _userFunc;      // CINT pointer to user function
                                 
  virtual RooCatType evaluate() const ;
  ClassDef(RooGenCategory,2) // Generic category-to-category function based on user supplied mapping function
};

#endif
