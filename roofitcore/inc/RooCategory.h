/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategory.rdl,v 1.15 2001/06/30 01:33:12 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_CATEGORY
#define ROO_CATEGORY

#include <iostream.h>
#include "RooFitCore/RooAbsCategoryLValue.hh"

class RooCategory : public RooAbsCategoryLValue {
public:
  // Constructor, assignment etc.
  RooCategory() {} ;
  RooCategory(const char *name, const char *title);
  RooCategory(const RooCategory& other, const char* name=0) ;
  virtual ~RooCategory();
  virtual TObject* clone(const char* newname) const { return new RooCategory(*this,newname); }
  virtual RooCategory& operator=(const RooCategory& other) ; 

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

protected:

  virtual RooCatType evaluate() const { return RooCatType() ;} // dummy because we overload getIndex()/getLabel()

  ClassDef(RooCategory,1) // Index variable 
};

#endif
