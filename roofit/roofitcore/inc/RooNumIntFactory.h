/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumIntFactory.h,v 1.6 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_NUM_INT_FACTORY
#define ROO_NUM_INT_FACTORY

#include <map>
#include <string>
#include "TObject.h"
#include "RooLinkedList.h"
#include "RooAbsIntegrator.h"
class RooNumIntConfig ;
class RooAbsFunc ;

class RooNumIntFactory ;
typedef void (*RooNumIntInitializerFunc)(RooNumIntFactory&) ;

class RooNumIntFactory : public TObject {
public:

  static RooNumIntFactory& instance() ;
  ~RooNumIntFactory() override = default;

  Bool_t storeProtoIntegrator(RooAbsIntegrator* proto, const RooArgSet& defConfig, const char* depName="") ;
  const RooAbsIntegrator* getProtoIntegrator(const char* name) const;
  const char* getDepIntegratorName(const char* name) const;

  RooAbsIntegrator* createIntegrator(RooAbsFunc& func, const RooNumIntConfig& config, Int_t ndim=0, Bool_t isBinned=kFALSE) const;


private:

  friend class RooNumIntConfig ;

  std::map<std::string,std::pair<std::unique_ptr<RooAbsIntegrator>,std::string> > _map;

  RooNumIntFactory() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see ROOT-10300
  RooNumIntFactory(const RooNumIntFactory& other) = delete;

  void init();


  ClassDefOverride(RooNumIntFactory, 0) // Numeric Integrator factory
};

#endif


