/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCategoryLValue.h,v 1.22 2007/05/11 09:11:30 verkerke Exp $
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

#include "Riosfwd.h"
#include "RooAbsCategory.h"
#include "RooAbsLValue.h"

class RooAbsCategoryLValue : public RooAbsCategory, public RooAbsLValue {
public:
  // Constructor, assignment etc.
  RooAbsCategoryLValue() {
    // Default constructor
  } ;
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
  virtual void setBin(Int_t ibin, const char* rangeName=0) ;
  virtual Int_t getBin(const char* rangeName=0) const ;
  virtual Int_t numBins(const char* rangeName) const ;
  virtual Double_t getBinWidth(Int_t /*i*/, const char* /*rangeName*/=0) const { 
    // Return volume of i-th bin (according to binning named rangeName if rangeName!=0)
    return 1.0 ; 
  }
  virtual Double_t volume(const char* rangeName) const { 
    // Return span of range with given name (=number of states included in this range)
    return numTypes(rangeName) ; 
  }
  virtual void randomize(const char* rangeName=0);

  virtual const RooAbsBinning* getBinningPtr(const char* /*rangeName*/) const { return 0 ; }
  virtual Int_t getBin(const RooAbsBinning* /*ptr*/) const { return getBin((const char*)0) ; }


  inline void setConstant(Bool_t value= kTRUE) { 
    // Declare category constant 
    setAttribute("Constant",value); 
  }
  
  inline virtual Bool_t isLValue() const { 
    // Object is an l-value
    return kTRUE; 
  }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  friend class RooSimGenContext ;
  virtual void setIndexFast(Int_t index) { _value._value = index ; _value._label[0]=0 ; }

  Bool_t setOrdinal(UInt_t index, const char* rangeName);
  void copyCache(const RooAbsArg* source, Bool_t valueOnly=kFALSE, Bool_t setValDirty=kTRUE) ;

  ClassDef(RooAbsCategoryLValue,1) // Abstract modifiable index variable 
};

#endif
