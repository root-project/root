/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAICRegistry.rdl,v 1.2 2001/10/13 21:53:18 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Sep-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_AIC_REGISTRY
#define ROO_AIC_REGISTRY

#include <iostream.h>
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

  Int_t _regSize ;
  Int_t* _clSize;       //! do not persist
  pInt_t* _clArr;       //! do not persist
  pRooArgSet* _asArr1;  //! do not persist
  pRooArgSet* _asArr2;  //! do not persist
  pRooArgSet* _asArr3;  //! do not persist
  pRooArgSet* _asArr4;  //! do not persist

  ClassDef(RooAICRegistry,1) 
} ;

#endif 
