/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategoryLValue.rdl,v 1.5 2001/05/31 21:21:35 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_CATEGORY_LVALUE
#define ROO_ABS_CATEGORY_LVALUE

#include <iostream.h>
#include "RooFitCore/RooAbsCategory.hh"

class RooAbsCategoryLValue : public RooAbsCategory {
public:
  // Constructor, assignment etc.
  RooAbsCategoryLValue() {} ;
  RooAbsCategoryLValue(const char *name, const char *title);
  RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name=0) ;
  virtual ~RooAbsCategoryLValue();

  // Value modifiers
  virtual Bool_t setIndex(Int_t index, Bool_t printError=kTRUE) = 0 ;
  virtual Bool_t setLabel(const char* label, Bool_t printError=kTRUE) = 0 ;
  RooAbsCategoryLValue& operator=(int index) ; 
  RooAbsCategoryLValue& operator=(const char* label) ; 
  void randomize();

  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }
  
  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  Bool_t setOrdinal(UInt_t index);
  void copyCache(const RooAbsArg* source) ;

  ClassDef(RooAbsCategoryLValue,1) // Abstract modifiable index variable 
};

#endif
