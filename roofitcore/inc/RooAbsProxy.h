/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsProxy.rdl,v 1.2 2001/05/14 22:54:19 verkerke Exp $
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

class RooAbsProxy : public TNamed {
public:

  // Constructors, assignment etc.
  RooAbsProxy() {} ;
  RooAbsProxy(const char* name, const char* desc, Bool_t valueServer, Bool_t shapeServer) ;
  RooAbsProxy(const char* name, const RooAbsProxy& other) ;
  virtual ~RooAbsProxy() {} ;

protected:

  RooDataSet* _dset ;
  Bool_t _valueServer ;
  Bool_t _shapeServer ;
  friend class RooAbsArg ;
  friend class RooAbsPdf ;

  inline Bool_t isValueServer() const { return _valueServer ; }
  inline Bool_t isShapeServer() const { return _shapeServer ; }
  virtual Bool_t changePointer(const RooArgSet& newServerSet) = 0 ;
  void changeDataSet(const RooDataSet* newDataSet) ;

  ClassDef(RooAbsProxy,0) // Abstract proxy interface
} ;

#endif

