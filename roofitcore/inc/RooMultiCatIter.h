/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMultiCatIter.rdl,v 1.5 2001/08/02 21:39:10 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_MULTI_CAT_ITER
#define ROO_MULTI_CAT_ITER

#include <iostream.h>
#include "TIterator.h"
#include "RooFitCore/RooArgSet.hh"
#include "TObjString.h"
class RooCategory ;
class RooCatType ;

typedef TIterator* pTIterator ;
typedef RooAbsCategoryLValue* pRooCategory ;

class RooMultiCatIter : public TIterator {
public:
  // Constructors, assignment etc.
  RooMultiCatIter(const RooArgSet& catList) ;
  RooMultiCatIter(const RooMultiCatIter& other) ;
  virtual ~RooMultiCatIter() ;

  // Iterator implementation
  virtual const TCollection* GetCollection() const ;
  virtual TObject* Next() ;
  virtual void Reset() ;

protected:
  void initialize(const RooArgSet& catList) ;
  TObjString* compositeLabel() ;

  RooArgSet        _catList  ;   // Set of categories iterated over
  pTIterator*      _iterList ;   // Array of category type iterators 
  pRooCategory*  _catPtrList ;   // Array of pointers to original categories
  RooCatType*   _curTypeList ;   // List of current types
  Int_t _nIter ;                 // Number of categories/iterators in use
  Int_t _curIter ;               // Current location of master iterator
  TObjString _compositeLabel ;

  ClassDef(RooMultiCatIter,0) // Iterator over all state permutations of a list of categories
};

#endif
