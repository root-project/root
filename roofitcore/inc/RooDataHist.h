/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataHist.rdl,v 1.9 2001/11/22 01:07:10 verkerke Exp $
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
#include "RooFitCore/RooDirItem.hh"
#include "RooFitCore/RooArgSet.hh"

class RooAbsArg;
class RooAbsReal ;
class RooAbsCategory ;
class Roo1DTable ;
class RooPlot;
class RooFitContext ;
class RooArgSet ;

class RooDataHist : public RooTreeData, public RooDirItem {
public:

  // Constructors, factory methods etc.
  RooDataHist() ; 
  RooDataHist(const char *name, const char *title, const RooArgSet& vars) ;
  RooDataHist(const char *name, const char *title, const RooArgSet& vars, const RooAbsData& data, Double_t initWgt=1.0) ;
  RooDataHist(const char *name, const char *title, const RooArgList& vars, const TH1* hist, Double_t initWgt=1.0) ;
  RooDataHist(const RooDataHist& other, const char* newname = 0) ;
  virtual TObject* Clone(const char* newname=0) const { return new RooDataHist(*this,newname?newname:GetName()) ; }
  virtual ~RooDataHist() ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1.0) ;
  void add(const RooAbsData& dset, const RooFormulaVar* cutVar=0, Double_t weight=1.0 ) ;
  void add(const RooAbsData& dset, const char* cut, Double_t weight=1.0 ) ;
  virtual const RooArgSet* get() const { return &_vars ; } 
  virtual const RooArgSet* get(Int_t masterIdx) const ;
  virtual const RooArgSet* get(const RooArgSet& coord) const ;
  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const ; 

  Double_t sum(Bool_t correctForBinSize) const ;
  Double_t sum(const RooArgSet& sumSet, const RooArgSet& sliceSet, Bool_t correctForBinSize) ;

  virtual Double_t weight() const { return _curWeight ; }
  Double_t weight(const RooArgSet& bin, Int_t intOrder=1) ; 

  virtual void reset() ;
  void dump2() ;

protected:
 
  void initialize() ;
  RooDataHist(const char* name, const char* title, RooDataHist* h, const RooArgSet& varSubset, 
	      const RooFormulaVar* cutVar, Bool_t copyCache) ;
  virtual RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache=kTRUE) ;
  Double_t interpolateDim(RooRealVar& dim, Double_t xval, Int_t intOrder) ;


  Int_t calcTreeIndex() const ;

  Int_t       _arrSize ; //  Size of the weight array
  Int_t*      _idxMult ; //! Multiplier jump table for index calculation
  Double_t*       _wgt ; //[_arrSize]  Weight array
  RooArgSet  _realVars ; // Real dimensions of the dataset 
  TIterator* _realIter ; //! Iterator over realVars

  mutable Double_t _curWeight ; // Weight associated with the current coordinate

private:

  ClassDef(RooDataHist,1) // Binned data set
};

#endif

