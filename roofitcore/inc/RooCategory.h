/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCategory.rdl,v 1.25 2005/12/08 13:19:54 wverkerke Exp $
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
#ifndef ROO_CATEGORY
#define ROO_CATEGORY

#include "Riostream.h"
#include "RooFitCore/RooAbsCategoryLValue.hh"

#include "RooFitCore/RooSharedPropertiesList.hh"
#include "RooFitCore/RooCategorySharedProperties.hh"

class RooCategory : public RooAbsCategoryLValue {
public:
  // Constructor, assignment etc.
  RooCategory() ;
  RooCategory(const char *name, const char *title);
  RooCategory(const RooCategory& other, const char* name=0) ;
  virtual ~RooCategory();
  virtual TObject* clone(const char* newname) const { return new RooCategory(*this,newname); }

  // Value modifiers
  virtual Int_t getIndex() const { return _value.getVal() ; }
  virtual const char* getLabel() const { return _value.GetName() ; }
  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) ;
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline virtual Bool_t isFundamental() const { return kTRUE; }

  Bool_t defineType(const char* label) ;
  Bool_t defineType(const char* label, Int_t index) ;
  void clearTypes() { RooAbsCategory::clearTypes() ; }

  void clearRange(const char* name, Bool_t silent) ;
  void setRange(const char* rangeName, const char* stateNameList) ;
  void addToRange(const char* rangeName, const char* stateNameList) ;
  Bool_t isStateInRange(const char* rangeName, const char* stateName) const ;
  virtual Bool_t inRange(const char* rangeName) const { return isStateInRange(rangeName,getLabel()) ; } 
  virtual Bool_t hasRange(const char* rangeName) const { return _sharedProp->_altRanges.FindObject(rangeName) ? kTRUE : kFALSE ; }
 
protected:

  static RooSharedPropertiesList _sharedPropList; // List of properties shared among clone sets 
  RooCategorySharedProperties* _sharedProp ; // Shared properties associated with this instance

  virtual RooCatType evaluate() const { return RooCatType() ;} // dummy because we overload getIndex()/getLabel()

  ClassDef(RooCategory,1) // Index variable 
};

#endif
