/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategory.rdl,v 1.10 2001/05/10 00:16:07 verkerke Exp $
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
  virtual TObject* clone() const { return new RooCategory(*this); }
  virtual RooCategory& operator=(const RooCategory& other) ; 

  // Value modifiers
  virtual Int_t getIndex() const { return _value.getVal() ; }
  virtual const char* getLabel() const { return _value.GetName() ; }
  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) ;
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  Bool_t defineType(const char* label) { return RooAbsCategory::defineType(label)?kFALSE:kTRUE ; }
  Bool_t defineType(const char* label, Int_t index) { return RooAbsCategory::defineType(label,index)?kFALSE:kTRUE ; }
  void clearTypes() { RooAbsCategory::clearTypes() ; }

protected:

  virtual RooCatType evaluate() const {} // dummy because we overload getIndex()/getLabel()

  ClassDef(RooCategory,1) // a real-valued variable and its value
};

#endif
