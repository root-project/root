/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooIntegratorConfig.rdl,v 1.7 2004/04/05 22:44:11 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NUM_INT_FACTORY
#define ROO_NUM_INT_FACTORY

#include "TObject.h"
#include "RooFitCore/RooLinkedList.hh"
#include "RooFitCore/RooAbsIntegrator.hh"
class RooNumIntConfig ;
class RooAbsFunc ;

class RooNumIntFactory : public TObject {
public:

  static RooNumIntFactory& instance() ;
  virtual ~RooNumIntFactory();

  Bool_t storeProtoIntegrator(RooAbsIntegrator* proto, const RooArgSet& defConfig, const char* depName="") ;
  const RooAbsIntegrator* getProtoIntegrator(const char* name) const ;
  const char* getDepIntegratorName(const char* name) const ;

  RooAbsIntegrator* createIntegrator(RooAbsFunc& func, const RooNumIntConfig& config, Int_t ndim=0) const ;

protected:

  static RooNumIntFactory* _instance ;
  RooLinkedList _integratorList ; // List of integrator prototypes
  RooLinkedList _nameList ;       // List of integrator names
  RooLinkedList _depList ;        // List of dependent integrator names

  RooNumIntFactory(); 
  RooNumIntFactory(const RooNumIntFactory& other) ;

  ClassDef(RooNumIntFactory,1) // Numeric Integrator factory
};

#endif


