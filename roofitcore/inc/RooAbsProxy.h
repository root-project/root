/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsProxy.rdl,v 1.7 2001/08/02 21:39:07 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_PROXY
#define ROO_ABS_PROXY

#include "TObject.h"
#include "RooFitCore/RooAbsArg.hh"

class RooAbsProxy {
public:

  // Constructors, assignment etc.
  RooAbsProxy() ;
  RooAbsProxy(const char* name, const RooAbsProxy& other) ;
  virtual ~RooAbsProxy() {} ;

  virtual const char* name() const { return "dummy" ; } ;  
  inline const RooArgSet* nset() const { return _nset ; }

protected:

  RooArgSet* _nset ;

  friend class RooAbsArg ;
  virtual Bool_t changePointer(const RooAbsCollection& newServerSet, Bool_t nameChange=kFALSE) = 0 ;

  friend class RooAbsPdf ;
  virtual void changeNormSet(const RooArgSet* newNormSet) ;

  ClassDef(RooAbsProxy,0) // Abstract proxy interface
} ;

#endif

