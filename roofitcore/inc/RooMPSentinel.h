/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/
#ifndef ROO_MP_SENTINEL
#define ROO_MP_SENTINEL

#include "Rtypes.h"
#include "RooFitCore/RooArgSet.hh"
class RooRealMPFE ;

class RooMPSentinel {
public:

  RooMPSentinel() ;
  ~RooMPSentinel() ;
 
protected:

  friend class RooRealMPFE ;
  void add(RooRealMPFE& mpfe) ;
  void remove(RooRealMPFE& mpfe) ;

  RooMPSentinel(const RooMPSentinel&) {}
  RooArgSet _mpfeSet ;
  
  ClassDef(RooMPSentinel,1) // Singleton class that terminate MP server processes when parent exists
};

#endif
