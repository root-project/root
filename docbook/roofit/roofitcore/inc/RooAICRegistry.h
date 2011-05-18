/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAICRegistry.h,v 1.11 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_AIC_REGISTRY
#define ROO_AIC_REGISTRY

#include "Riosfwd.h"
#include <assert.h>
#include "Rtypes.h"
class RooArgSet ;

typedef Int_t* pInt_t ;
typedef RooArgSet* pRooArgSet ;

class RooAICRegistry {

public:
  RooAICRegistry(Int_t regSize=10) ;
  RooAICRegistry(const RooAICRegistry& other) ;
  virtual ~RooAICRegistry() ;

  Int_t store(Int_t* codeList, Int_t size, RooArgSet* set1=0, RooArgSet* set2=0, RooArgSet* set3=0, RooArgSet* set4=0) ;
  const Int_t* retrieve(Int_t masterCode) const ;
  const Int_t* retrieve(Int_t masterCode, pRooArgSet& set1) const ;
  const Int_t* retrieve(Int_t masterCode, pRooArgSet& set1, pRooArgSet& set2) const ;
  const Int_t* retrieve(Int_t masterCode, pRooArgSet& set1, pRooArgSet& set2, pRooArgSet& set3, pRooArgSet& set4) const ;

protected:

  Int_t _regSize ;      // Size of registry
  Int_t* _clSize;       //! Array is size of code lists
  pInt_t* _clArr;       //! Array of array of code lists
  pRooArgSet* _asArr1;  //! Array of 1st RooArgSet pointers
  pRooArgSet* _asArr2;  //! Array of 2nd RooArgSet pointers
  pRooArgSet* _asArr3;  //! Array of 3rd RooArgSet pointers
  pRooArgSet* _asArr4;  //! Array of 4th RooArgSet pointers

  ClassDef(RooAICRegistry,1) // Registry for analytical integration codes
} ;

#endif 
