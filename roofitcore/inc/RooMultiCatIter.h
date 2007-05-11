/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMultiCatIter.rdl,v 1.13 2005/06/20 15:44:55 wverkerke Exp $
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
#ifndef ROO_MULTI_CAT_ITER
#define ROO_MULTI_CAT_ITER

#include "Riostream.h"
#include "TIterator.h"
#include "RooArgSet.h"
#include "TObjString.h"
class RooCategory ;
class RooCatType ;

typedef TIterator* pTIterator ;
typedef RooAbsCategoryLValue* pRooCategory ;

class RooMultiCatIter : public TIterator {
public:
  // Constructors, assignment etc.
  RooMultiCatIter(const RooArgSet& catList, const char* rangeName=0) ;
  RooMultiCatIter(const RooMultiCatIter& other) ;
  virtual ~RooMultiCatIter() ;

  // Iterator implementation
  virtual const TCollection* GetCollection() const ;
  virtual TObject* Next() ;
  virtual void Reset() ;

protected:
  
  TIterator& operator=(const TIterator&) { return *this ; } // forbidden for now

  void initialize(const RooArgSet& catList) ;
  TObjString* compositeLabel() ;

  RooArgSet        _catList  ;   // Set of categories iterated over
  pTIterator*      _iterList ;   // Array of category type iterators 
  pRooCategory*  _catPtrList ;   // Array of pointers to original categories
  RooCatType*   _curTypeList ;   // List of current types
  Int_t _nIter ;                 // Number of categories/iterators in use
  Int_t _curIter ;               // Current location of master iterator
  TObjString _compositeLabel ;   //
  TString _rangeName ;           // Range name (optional)

  ClassDef(RooMultiCatIter,0) // Iterator over all state permutations of a list of categories
};

#endif
