/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataHist.rdl,v 1.3 2001/09/17 18:48:13 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_DATA_HIST
#define ROO_DATA_HIST

#include "TObject.h"
#include "RooFitCore/RooTreeData.hh"
#include "RooFitCore/RooArgSet.hh"

class RooAbsArg;
class RooAbsReal ;
class RooAbsCategory ;
class Roo1DTable ;
class RooPlot;
class RooFitContext ;
class RooHistArray ;

class RooDataHist : public RooTreeData {
public:

  // Constructors, factory methods etc.
  RooDataHist() ; 
  RooDataHist(const char *name, const char *title, const RooArgSet& vars) ;
  RooDataHist(const char *name, const char *title, const RooArgSet& vars, const RooAbsData& data) ;
  RooDataHist(const RooDataHist& other, const char* newname = 0) ;
  virtual TObject* Clone(const char* newname=0) const { return new RooDataHist(*this,newname?newname:GetName()) ; }
  virtual ~RooDataHist() ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1.0) ;
  void add(const RooAbsData& dset, const RooFormulaVar* cutVar=0, Double_t weight=1.0 ) ;
  void add(const RooAbsData& dset, const char* cut, Double_t weight=1.0 ) ;
  virtual const RooArgSet* get() const { return &_vars ; } 
  virtual const RooArgSet* get(Int_t masterIdx) const ;
  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const ; 

  Double_t sum(Bool_t correctForBinSize) const ;
  Double_t sum(const RooArgSet& sumSet, const RooArgSet& sliceSet, Bool_t correctForBinSize) ;

  virtual Double_t weight() const { return _curWeight ; }
  Double_t weight(const RooArgSet& bin) ; 

  virtual void reset() ;
  void dump2() ;

protected:
 
  void initialize() ;
  RooDataHist(const char* name, const char* title, RooDataHist* h, const RooArgSet& varSubset, 
	      const RooFormulaVar* cutVar, Bool_t copyCache) ;
  virtual RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache=kTRUE) ;


  Int_t calcTreeIndex() const ;

  Int_t     _arrSize ;
  Int_t*    _idxMult ; //! do not persist
  Double_t*     _wgt ; //[_arrSize]  Weight array

  mutable Double_t _curWeight ;

private:

  ClassDef(RooDataHist,1) // Binned data set
};

#endif

