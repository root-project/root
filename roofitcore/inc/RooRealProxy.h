/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStreamParser.rdl,v 1.2 2001/03/22 15:31:25 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_REAL_PROXY
#define ROO_REAL_PROXY

#include "RooFitCore/RooAbsReal.hh"

class RooRealProxy {
public:

  // Constructors, assignment etc.
  RooRealProxy(RooAbsReal& ref) ;
  virtual ~RooRealProxy();

  // Accessors
  operator Double_t() { return _ref.getVal() ; }
  const RooAbsReal& arg() { return _ref ; }

protected:

  RooAbsReal& _ref ;  
  ClassDef(RooRealProxy,0) // not persistable 
};

#endif
