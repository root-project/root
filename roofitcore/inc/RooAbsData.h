/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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

class RooAbsArg;
class RooAbsReal ;
class RooAbsCategory ;
class Roo1DTable ;
class RooPlot;
class RooFitContext ;

class RooAbsData : public TNamed, public RooPrintable {
public:

  // Constructors, factory methods etc.
  RooAbsData() ; 
  RooAbsData(const char *name, const char *title, const RooArgSet& vars) ;
  virtual ~RooAbsData() ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1) = 0 ;

  // Load a given row of data
  virtual inline const RooArgSet* get() const { return &_vars ; } // last loaded row
  virtual const RooArgSet* get(Int_t index) const = 0 ;

  // Plot the distribution of a real valued arg
  virtual Roo1DTable* table(RooAbsCategory& cat, const char* cuts="", const char* opts="") = 0;
  virtual RooPlot *plotOn(RooPlot *frame, const char* cuts="", Option_t* drawOptions="P") const = 0 ;
 
  // Printing interface (human readable)
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:

  // RooFitContext optimizer interface
  friend class RooFitContext ;
  virtual void cacheArg(RooAbsArg& var) = 0 ;
  virtual void cacheArgs(RooArgSet& varSet) = 0 ;
  void setDirtyProp(Bool_t flag) { _doDirtyProp = flag ; }

  // Column structure definition
  RooArgSet _vars;         
  RooArgSet _cachedVars ;  

  TIterator *_iterator;    //! don't make this data member persistent
  TIterator *_cacheIter ;  //! don't make this data member persistent
  Bool_t _doDirtyProp ;

private:

  ClassDef(RooAbsData,1) // Data set for fitting
};

#endif
