/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCatType.rdl,v 1.9 2001/05/14 22:54:20 verkerke Exp $
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
#include "TObject.h"
#include "RooFitCore/RooPrintable.hh"

class RooCatType : public TObject, public RooPrintable {
public:
  inline RooCatType() : TObject() { _value = 0 ; _label[0] = 0 ; } 
  inline RooCatType(const char* name, Int_t value) : TObject(), _value(value) { SetName(name) ; } 
  inline RooCatType(const RooCatType& other) : TObject(other), _value(other._value) { strcpy(_label,other._label) ;} ;
  virtual TObject* Clone(const char* newname=0) const { return new RooCatType(*this); }
  virtual const Text_t* GetName() const { return _label ; }
  virtual void SetName(const Text_t* name) { 
    if (strlen(name)>255) {
      cout << "RooCatType::SetName warning: label '" << name << "' truncated at 255 chars" << endl ;
      _label[255]=0 ;
    }
    strncpy(_label,name,255) ;
  }

  inline RooCatType& operator=(const RooCatType& other) 
    { SetName(other.GetName()) ; _value = other._value ; return *this ; } 

  inline Bool_t operator==(const RooCatType& other) {
    return ( _value==other._value && !strcmp(_label,other._label)) ;
  }

  inline Bool_t operator==(Int_t index) { return (_value==index) ; }

  Bool_t operator==(const char* label) { return !strcmp(_label,label) ; }

  inline Int_t getVal() const { return _value ; }
  void setVal(Int_t newValue) { _value = newValue ; }

  void printToStream(ostream& os, PrintOption opt= Standard, TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:
  friend class RooAbsCategory ;
  Int_t _value ;
  char _label[256] ;
	
  ClassDef(RooCatType,1) // Category state, (name,index) pair
} ;


#endif

