/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   25-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
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

  //protected:

  void constOptimize(ConstOpCode opcode) ;

  void optimizeDirty() ;
  void doConstOpt() ;
  void undoConstOpt() ;  

  // Prefit optimizer
  Bool_t findCacheableBranches(RooAbsArg* arg, RooAbsData* dset, RooArgSet& cacheList) ;
  void findUnusedDataVariables(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& pruneList) ;
  void findRedundantCacheServers(RooAbsPdf* pdf,RooAbsData* dset,RooArgSet& cacheList, RooArgSet& pruneList) ;
  Bool_t allClientsCached(RooAbsArg* var, RooArgSet& cacheList) ;

  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange) ;

  RooArgSet*  _normSet ;
  RooArgSet*  _pdfCloneSet ;
  RooAbsData* _dataClone ;
  RooAbsPdf*  _pdfClone ;
  RooArgSet*  _projDeps ;

  ClassDef(RooAbsOptGoodnessOfFit,1) // Abstract real-valued variable
};

#endif
