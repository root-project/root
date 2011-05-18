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
#include "TMethodCall.h"

class RooGenCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooGenCategory() { 
    // Default constructor
    // coverity[UNINIT_CTOR]
  }
  RooGenCategory(const char *name, const char *title, void* userFunc, RooArgSet& catList);
  RooGenCategory(const RooGenCategory& other, const char *name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooGenCategory(*this,newname); }
  virtual ~RooGenCategory();

  // Printing interface (human readable)
  virtual void printMultiline(ostream& os, Int_t content, Bool_t verbose=kFALSE, TString indent= "") const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  void initialize() ;
  TString evalUserFunc(RooArgSet *vars) ;
  void updateIndexList() ;
  
  RooSuperCategory _superCat ;      //  Super category of input categories
  RooCategoryProxy _superCatProxy ; // Proxy for super category
  Int_t *_map ;                     //! Super-index to generic-index map

  TString      _userFuncName ; // Name of user function
  TMethodCall* _userFunc;      // CINT pointer to user function
  Long_t _userArgs[1];         // Placeholder for user function arguments
                                 
  virtual RooCatType evaluate() const ; 
  ClassDef(RooGenCategory,1) // Generic category-to-category function based on user supplied mapping function
};

#endif
