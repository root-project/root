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

#include <functional>

class RooNumIntConfig ;
class RooAbsFunc ;

class RooNumIntFactory ;
typedef void (*RooNumIntInitializerFunc)(RooNumIntFactory&) ;

class RooNumIntFactory : public TObject {
public:

  RooNumIntFactory(const RooNumIntFactory& other) = delete;

  using Creator = std::function<std::unique_ptr<RooAbsIntegrator>(RooAbsFunc const& function, const RooNumIntConfig& config)>;

  static RooNumIntFactory& instance() ;

  bool registerPlugin(std::string const &name, Creator const &creator, const RooArgSet &defConfig, bool canIntegrate1D,
                    bool canIntegrate2D, bool canIntegrateND, bool canIntegrateOpenEnded, const char *depName = "");

  std::string getIntegratorName(RooAbsFunc& func, const RooNumIntConfig& config, int ndim=0, bool isBinned=false) const;
  std::unique_ptr<RooAbsIntegrator> createIntegrator(RooAbsFunc& func, const RooNumIntConfig& config, int ndim=0, bool isBinned=false) const;


private:

  friend class RooNumIntConfig ;

  struct PluginInfo {
    Creator creator;
    bool canIntegrate1D = false;
    bool canIntegrate2D = false;
    bool canIntegrateND = false;
    bool canIntegrateOpenEnded = false;
    std::string depName;
  };

  PluginInfo const* getPluginInfo(std::string const& name) const
  {
    auto item = _map.find(name);
    return item == _map.end() ? nullptr : &item->second;
  }

  std::map<std::string,PluginInfo> _map;

  RooNumIntFactory() {} // NOLINT: not allowed to use = default because of TObject::kIsOnHeap detection, see https://sft.its.cern.ch/jira/browse/ROOT-10300

  void init();


  ClassDefOverride(RooNumIntFactory, 0) // Numeric Integrator factory
};

#endif
