/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCategory.h,v 1.27 2007/05/11 09:11:30 verkerke Exp $
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

#include "Riosfwd.h"
#include "RooAbsCategoryLValue.h"

#include "RooSharedPropertiesList.h"
#include "RooCategorySharedProperties.h"

class RooCategory : public RooAbsCategoryLValue {
public:
  // Constructor, assignment etc.
  RooCategory() ;
  RooCategory(const char *name, const char *title);
  RooCategory(const RooCategory& other, const char* name=0) ;
  virtual ~RooCategory();
  virtual TObject* clone(const char* newname) const { return new RooCategory(*this,newname); }

  // Value modifiers
  virtual Int_t getIndex() const { 
    return _value.getVal() ; 
    // Return index value
  }
  
  virtual const char* getLabel() const { 
    const char* ret = _value.GetName() ;
    if (ret==0) {
      _value.SetName(lookupType(_value.getVal())->GetName()) ;    
    }
    return _value.GetName() ;
  }

  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) ;
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline virtual Bool_t isFundamental() const { 
    // Return true as a RooCategory is a fundamental (non-derived) type
    return kTRUE; 
  }

  virtual Bool_t isDerived() const { 
    // Does value or shape of this arg depend on any other arg?
    return kFALSE ;
  }

  Bool_t defineType(const char* label) ;
  Bool_t defineType(const char* label, Int_t index) ;
  void clearTypes() { RooAbsCategory::clearTypes() ; }

  void clearRange(const char* name, Bool_t silent) ;
  void setRange(const char* rangeName, const char* stateNameList) ;
  void addToRange(const char* rangeName, const char* stateNameList) ;
  Bool_t isStateInRange(const char* rangeName, const char* stateName) const ;
  virtual Bool_t inRange(const char* rangeName) const { 
    // Returns true of current value of category is inside given range
    return isStateInRange(rangeName,getLabel()) ; 
  } 
  virtual Bool_t hasRange(const char* rangeName) const { 
    // Returns true if category has range with given name
    return _sharedProp->_altRanges.FindObject(rangeName) ? kTRUE : kFALSE ; 
  }
 
protected:
  
  static RooSharedPropertiesList _sharedPropList; // List of properties shared among clone sets 
  static RooCategorySharedProperties _nullProp ; // Null property
  RooCategorySharedProperties* _sharedProp ; //! Shared properties associated with this instance

  virtual RooCatType evaluate() const { 
    // Dummy implementation
    return RooCatType() ;
  } 

  ClassDef(RooCategory,2) // Discrete valued variable type
};

#endif
