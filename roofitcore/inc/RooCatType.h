/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.rdl,v 1.4 2001/04/14 00:43:19 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_CAT_TYPE
#define ROO_CAT_TYPE

#include <iostream.h>
#include "TNamed.h"
#include "RooFitCore/RooPrintable.hh"

class RooCatType : public TNamed, public RooPrintable {
public:
  inline RooCatType() : TNamed() { _value = 0 ; } 
  inline RooCatType(const char* name, Int_t value) : TNamed(name,""), _value(value) {} ;
  inline RooCatType(const RooCatType& other) : TNamed(other), _value(other._value) {} ;
  virtual TObject* Clone() { return new RooCatType(*this); }
  

  inline RooCatType& operator=(const RooCatType& other) 
    { SetName(other.GetName()) ; _value = other._value ; return *this ; } 

  inline Bool_t operator==(const RooCatType& other) {
    return ((*this)== other.getVal() && (*this)== other.GetName());
  }

  inline Bool_t operator==(Int_t index) { return (_value==index) ; }

  Bool_t operator==(const char* label) ;

  inline Int_t getVal() const { return _value ; }
  void setVal(Int_t newValue) { _value = newValue ; }

  void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:
  friend class RooCategory ;
  Int_t _value ;
	
  ClassDef(RooCatType,1) // a real-valued variable and its value
} ;


#endif

