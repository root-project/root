/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooFitResult.rdl,v 1.9 2002/05/16 01:14:44 verkerke Exp $
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
#include "TObject.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooDirItem.hh"

class TMatrix ;
class RooArgSet ;
class RooArgList ;
class RooPlot;
typedef RooArgSet* pRooArgSet ;

class RooFitResult : public TNamed, public RooPrintable, public RooDirItem {
public:

  // Constructors, assignment etc.
  RooFitResult(const char* name=0, const char* title=0) ;
  virtual ~RooFitResult() ;

  static RooFitResult* lastMinuitFit() ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  // Accessors
  inline Int_t status() const { return _status ; }
  inline Double_t edm() const { return _edm ; }
  inline Double_t minNll() const { return _minNLL ; }
  inline const RooArgList& constPars() const { return *_constPars ; } 
  inline const RooArgList& floatParsInit() const { return *_initPars ; } 
  inline const RooArgList& floatParsFinal() const { return *_finalPars ; } 

  // Correlation matrix element and row accessors
  Double_t correlation(const RooAbsArg& par1, const RooAbsArg& par2) const {
    return correlation(par1.GetName(),par2.GetName()) ;
  }
  const RooArgList* correlation(const RooAbsArg& par) const {
    return correlation(par.GetName()) ;
  }

  Double_t correlation(const char* parname1, const char* parname2) const ;
  const RooArgList* correlation(const char* parname) const ;

  // Global correlation accessors
  Double_t globalCorr(const RooAbsArg& par) { return globalCorr(par.GetName()) ; }
  Double_t globalCorr(const char* parname) ;
  const RooArgList* globalCorr() ;


  // Add objects to a 2D plot
  inline RooPlot *plotOn(RooPlot *frame, const RooAbsArg &par1, const RooAbsArg &par2,
			 const char *options= "ME") const {
    return plotOn(frame,par1.GetName(),par2.GetName(),options);
  }
  RooPlot *plotOn(RooPlot *plot, const char *parName1, const char *parName2,
		  const char *options= "ME") const;

  // Generate random perturbations of the final parameters using the covariance matrix
  const RooArgList& randomizePars() const;

protected:
  
  RooFitResult(const RooFitResult& other) ;

  friend class RooFitContext ;
  void setConstParList(const RooArgList& list) ;
  void setInitParList(const RooArgList& list) ;
  void setFinalParList(const RooArgList& list) ;
  inline void setMinNLL(Double_t val) { _minNLL = val ; }
  inline void setEDM(Double_t val) { _edm = val ; }
  inline void setStatus(Int_t val) { _status = val ; }
  void fillCorrMatrix() ;

  Double_t correlation(Int_t row, Int_t col) const;
  Double_t covariance(Int_t row, Int_t col) const;

  Int_t    _status ;          // MINUIT status code
  Double_t _minNLL ;          // NLL at minimum
  Double_t _edm ;             // Estimated distance to minimum
  RooArgList* _constPars ;    // List of constant parameters
  RooArgList* _initPars ;     // List of floating parameters with initial values
  RooArgList* _finalPars ;    // List of floating parameters with final values
  RooArgList* _globalCorr ;   // List of global correlation coefficients
  TList       _corrMatrix ;   // Correlation matrix (list of RooArgLists)

  mutable RooArgList *_randomPars; //! List of floating parameters with most recent random perturbation applied
  mutable TMatrix *_Lt;            //! triangular matrix used for generate random perturbations

  ClassDef(RooFitResult,1) // Container class for fit result
};

#endif
