/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   17-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_FIT_RESULT
#define ROO_FIT_RESULT

#include <iostream.h>
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"

class RooArgSet ;
typedef RooArgSet* pRooArgSet ;

class RooFitResult : public RooPrintable {
public:

  // Constructors, assignment etc.
  RooFitResult() ;
  virtual ~RooFitResult() ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  // Accessors
  inline Double_t edm() const { return _edm ; }
  inline Double_t minNll() const { return _minNLL ; }
  inline const RooArgSet& constPars() const { return *_constPars ; } 
  inline const RooArgSet& floatParsInit() const { return *_initPars ; } 
  inline const RooArgSet& floatParsFinal() const { return *_finalPars ; } 

  // Correlation matrix element and row accessors
  Double_t correlation(const RooAbsArg& par1, const RooAbsArg& par2) const ;
  const RooArgSet* correlation(const RooAbsArg& par) const ;

protected:

  friend class RooFitContext ;
  void setConstParList(const RooArgSet& list) ;
  void setInitParList(const RooArgSet& list) ;
  void setFinalParList(const RooArgSet& list) ;
  inline void setMinNLL(Double_t val) { _minNLL = val ; }
  inline void setEDM(Double_t val) { _edm = val ; }
  void fillCorrMatrix() ;

  Double_t _minNLL ;
  Double_t _edm ;
  RooArgSet* _constPars ;
  RooArgSet* _initPars ;
  RooArgSet* _finalPars ;
  RooArgSet* _globalCorr ;
  pRooArgSet* _corrMatrix ;

  ClassDef(RooFitResult,0) // Iterator over all bins of a RooAbsArg
};

#endif
