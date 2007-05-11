/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHashTable.rdl,v 1.11 2005/02/25 14:22:57 wverkerke Exp $
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
#ifndef ROO_HASH_TABLE
#define ROO_HASH_TABLE

#include "TObject.h"
#include "TString.h"

class RooAbsArg ;
class RooLinkedList ;
class RooLinkedListElem ;
class RooSetPair ;
class RooArgSet ;

class RooHashTable : public TObject {
public:

  enum HashMethod { Pointer=0, Name=1, Intrinsic=2 } ;

  // Constructor
  RooHashTable(Int_t initSize = 17, HashMethod hashMethod=Name) ;
  RooHashTable(const RooHashTable& other) ;

  // Destructor
  virtual ~RooHashTable() ;

  void add(TObject* arg, TObject* hashArg=0) ;
  Bool_t remove(TObject* arg, TObject* hashArg=0) ;
  TObject* find(const char* name) const ;
  TObject* find(const TObject* arg) const ;
  RooLinkedListElem* findLinkTo(const TObject* arg) const ;
  RooSetPair* findSetPair(const RooArgSet* set1, const RooArgSet* set2) const ;  
  Bool_t replace(const TObject* oldArg, const TObject* newArg, const TObject* oldHashArg=0) ;
  Int_t size() const { return _size ; }
  Double_t avgCollisions() const ;

protected:  
  inline ULong_t hash(const TObject* arg) const {
    switch(_hashMethod) {
      case Pointer:   return TString::Hash((void*)(&arg),sizeof(void*)) ;
      case Name:      return TString::Hash(arg->GetName(),strlen(arg->GetName())) ;
      case Intrinsic: return arg->Hash() ;
    }
    return 0 ;
  }

  HashMethod _hashMethod ;
  Int_t _usedSlots ;
  Int_t _entries ;
  Int_t _size ;
  RooLinkedList** _arr ; //! do not persist

  ClassDef(RooHashTable,1) // Hash table
};




#endif
