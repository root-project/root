/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
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
