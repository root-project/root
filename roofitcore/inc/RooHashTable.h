/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHashTable.rdl,v 1.7 2004/04/05 22:44:11 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_HASH_TABLE
#define ROO_HASH_TABLE

#include "TObject.h"
#include "TMath.h"

class RooAbsArg ;
class RooLinkedList ;

class RooHashTable : public TObject {
public:
  // Constructor
  RooHashTable(Int_t initSize = 17, Bool_t hashByPtr=kFALSE) ;
  RooHashTable(const RooHashTable& other) ;

  // Destructor
  virtual ~RooHashTable() ;

  void add(TObject* arg) ;
  Bool_t remove(TObject* arg) ;
  TObject* find(const char* name) const ;
  TObject* find(const TObject* arg) const ;
  Bool_t replace(const TObject* oldArg, const TObject* newArg) ;
  Int_t size() const { return _size ; }
  Double_t avgCollisions() const ;

protected:  
  inline ULong_t hash(const TObject* arg) const {
    return _hashByPtr ? TMath::Hash((void*)(&arg),sizeof(void*)) : arg->Hash() ;
  }

  Bool_t _hashByPtr ;
  Int_t _usedSlots ;
  Int_t _entries ;
  Int_t _size ;
  RooLinkedList** _arr ; //! do not persist

  ClassDef(RooHashTable,1) // Hash table
};




#endif
