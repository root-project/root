/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsOptGoodnessOfFit.rdl,v 1.6 2004/03/31 01:37:39 wverkerke Exp $
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
#ifndef ROO_ABS_OPT_GOODNESS_OF_FIT
#define ROO_ABS_OPT_GOODNESS_OF_FIT

#include "RooFitCore/RooAbsGoodnessOfFit.hh"
#include "RooFitCore/RooSetProxy.hh"

class RooArgSet ;
class RooAbsData ;
class RooAbsPdf ;

class RooAbsOptGoodnessOfFit : public RooAbsGoodnessOfFit {
public:

  // Constructors, assignment etc
  inline RooAbsOptGoodnessOfFit() { }
  RooAbsOptGoodnessOfFit(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		      const RooArgSet& projDeps, Int_t nCPU=1) ;
  RooAbsOptGoodnessOfFit(const RooAbsOptGoodnessOfFit& other, const char* name=0);
  virtual ~RooAbsOptGoodnessOfFit();

  virtual Double_t combinedValue(RooAbsReal** gofArray, Int_t nVal) const ;

protected:

  void constOptimize(ConstOpCode opcode) ;

  void optimizeDirty() ;
  void doConstOpt() ;
  void undoConstOpt() ;  

  // Prefit optimizer
  Bool_t findCacheableBranches(RooAbsArg* arg, RooAbsData* dset, RooArgSet& cacheList) ;
  void findUnusedDataVariables(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& pruneList) ;
  void findRedundantCacheServers(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& cacheList, RooArgSet& pruneList) ;
  Bool_t allClientsCached(RooAbsArg* var, RooArgSet& cacheList) ;

  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;
  virtual void printCompactTreeHook(const char* indent="") ;

  RooArgSet*  _normSet ;
  RooArgSet*  _pdfCloneSet ;
  RooAbsData* _dataClone ;
  RooAbsPdf*  _pdfClone ;
  RooArgSet*  _projDeps ;

  ClassDef(RooAbsOptGoodnessOfFit,1) // Abstract real-valued variable
};

#endif
