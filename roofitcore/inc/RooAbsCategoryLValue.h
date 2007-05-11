/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCategoryLValue.rdl,v 1.21 2005/06/23 15:08:55 wverkerke Exp $
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
#ifndef ROO_ABS_CATEGORY_LVALUE
#define ROO_ABS_CATEGORY_LVALUE

#include "Riostream.h"
#include "RooAbsCategory.h"
#include "RooAbsLValue.h"

class RooAbsCategoryLValue : public RooAbsCategory, public RooAbsLValue {
public:
  // Constructor, assignment etc.
  RooAbsCategoryLValue() {} ;
  RooAbsCategoryLValue(const char *name, const char *title);
  RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name=0) ;
  virtual ~RooAbsCategoryLValue();

  // Value modifiers
  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) = 0 ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) = 0 ;
  RooAbsArg& operator=(int index) ; 
  RooAbsArg& operator=(const char* label) ; 
  RooAbsArg& operator=(const RooAbsCategory& other) ;

  // Binned fit interface
  virtual void setBin(Int_t ibin) ;
  virtual Int_t getBin() const ;
  virtual Int_t numBins() const ;
  virtual Double_t getBinWidth(Int_t /*i*/) const { return 1.0 ; }

  virtual void randomize();
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }
  
  inline virtual Bool_t isLValue() const { return kTRUE; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  Bool_t setOrdinal(UInt_t index);
  void copyCache(const RooAbsArg* source) ;

  ClassDef(RooAbsCategoryLValue,1) // Abstract modifiable index variable 
};

#endif
