/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.rdl,v 1.1 2001/03/17 00:32:54 verkerke Exp $
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

class RooCatType : public TNamed {
public:
  inline RooCatType() : TNamed() { _value = 0 ; } 
  inline RooCatType(const char* name, Int_t value) : TNamed(name,""), _value(value) {} ;
  inline RooCatType(const RooCatType& other) : TNamed(other), _value(other._value) {} ;

  inline RooCatType& operator=(const RooCatType& other) { SetName(other.GetName()) ; _value = other._value ; return *this ; } 
  inline Bool_t operator==(const RooCatType& comp) { return (_value==comp._value) ; }
  inline Bool_t operator==(Int_t index) { return (_value==index) ; }
  inline Bool_t operator==(const char* label) { return !TString(label).CompareTo(GetName()) ; }
  inline operator Int_t&() { return _value ; }
  inline operator Int_t() const { return _value ; }
  inline Int_t getVal() const { return _value ; }
  void setVal(Int_t newValue) { _value = newValue ; }

  enum PrintOption { Standard=0 } ;
  void printToStream(ostream& os, PrintOption opt=Standard) 
       { os << GetName() << ":" << _value << endl ; }
  void print(PrintOption opt=Standard) { printToStream(cout,opt) ; }

protected:
  Int_t _value ;
	
  ClassDef(RooCatType,1) // a real-valued variable and its value
} ;


#endif

