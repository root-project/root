/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsData.rdl,v 1.14 2002/02/20 19:46:21 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_DATA
#define ROO_ABS_DATA

#include "TNamed.h"
#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooFormulaVar.hh"

class RooAbsArg;
class RooAbsReal ;
class RooAbsCategory ;
class Roo1DTable ;
class RooPlot;
class RooFitContext ;
class RooArgList;
class TH1;

class RooAbsData : public TNamed, public RooPrintable {
public:

  // Constructors, factory methods etc.
  RooAbsData() ; 
  RooAbsData(const char *name, const char *title, const RooArgSet& vars) ;
  RooAbsData(const RooAbsData& other, const char* newname = 0) ;
  virtual ~RooAbsData() ;
  virtual RooAbsData* emptyClone(const char* newName=0, const char* newTitle=0) const = 0 ;

  // Reduction methods
  RooAbsData* reduce(const char* cut) ;
  RooAbsData* reduce(const RooFormulaVar& cutVar) ;
  RooAbsData* reduce(const RooArgSet& varSubset, const char* cut=0) ;
  RooAbsData* reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar) ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1) = 0 ;
  virtual void fill() = 0 ;

  // Load a given row of data
  virtual inline const RooArgSet* get() const { return &_vars ; } // last loaded row
  virtual Double_t weight() const = 0 ; 
  virtual const RooArgSet* get(Int_t index) const = 0 ;

  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const = 0 ;
  virtual void reset() = 0 ;

  // Plot the distribution of a real valued arg
  virtual Roo1DTable* table(const RooAbsCategory& cat, const char* cuts="", const char* opts="") const = 0;
  virtual RooPlot *plotOn(RooPlot *frame, const char* cuts="", Option_t* drawOptions="P") const = 0 ;
  virtual RooPlot *plotOn(RooPlot *frame, const RooFormulaVar *cutVar, Option_t* drawOptions="P") const = 0 ;
  virtual RooPlot *plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, 
			      const char* cut="", Option_t* drawOptions="P") const = 0 ;

  // Split a dataset by a category
  virtual TList* split(const RooAbsCategory& splitCat) const = 0 ;
 
  // Fill an existing histogram
  virtual TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "") const = 0;

  // Printing interface (human readable)
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:

  // RooFitContext optimizer interface
  friend class RooFitContext ;
  friend class RooSimFitContext ;
  virtual void cacheArgs(RooArgSet& varSet, const RooArgSet* nset=0) = 0 ;
  void setDirtyProp(Bool_t flag) { _doDirtyProp = flag ; }

  virtual RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, Bool_t copyCache=kTRUE) = 0 ;

  // Column structure definition
  RooArgSet _vars;         // Dimensions of this data set
  RooArgSet _cachedVars ;  //! External variables cached with this data set

  TIterator *_iterator;    //! Iterator over dimension variables
  TIterator *_cacheIter ;  //! Iterator over cached variables
  Bool_t _doDirtyProp ;    // Switch do (de)activate dirty state propagation when loading a data point

private:

  ClassDef(RooAbsData,1) // Abstract data collection
};

#endif
