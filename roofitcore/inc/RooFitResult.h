/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooFitResult.rdl,v 1.26 2006/08/19 00:39:58 bondioli Exp $
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
#ifndef ROO_FIT_RESULT
#define ROO_FIT_RESULT

#include "Riostream.h"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooDirItem.hh"
#include "RooFitCore/RooArgList.hh"

#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,00,00)
#include "TMatrixFfwd.h"
#include "TRootIOCtor.h"
#else
class TMatrixF;
#endif

class RooArgSet ;
class RooPlot;
class TObject ;
typedef RooArgSet* pRooArgSet ;

class RooFitResult : public TNamed, public RooPrintable, public RooDirItem {
public:
 
  // Constructors, assignment etc.
  RooFitResult(const char* name=0, const char* title=0) ;
  RooFitResult(const RooFitResult& other) ;   				// added, FMV 08/13/03
  virtual TObject* clone() const { return new RooFitResult(*this); }   	// added, FMV 08/13/03
  virtual ~RooFitResult() ;

  static RooFitResult* lastMinuitFit(const RooArgList& varList=RooArgList()) ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

  // Accessors
  inline Int_t status() const { return _status ; }
  inline Int_t covQual() const { return _covQual ; }
  inline Int_t numInvalidNLL() const { return _numBadNLL ; }
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
  
  friend class RooMinuit ;
  friend class RooNag ;
  void setConstParList(const RooArgList& list) ;
  void setInitParList(const RooArgList& list) ;
  void setFinalParList(const RooArgList& list) ;
  inline void setMinNLL(Double_t val) { _minNLL = val ; }
  inline void setEDM(Double_t val) { _edm = val ; }
  inline void setStatus(Int_t val) { _status = val ; }
  inline void setCovQual(Int_t val) { _covQual = val ; }
  inline void setNumInvalidNLL(Int_t val) { _numBadNLL=val ; }
  void fillCorrMatrix() ;

  Double_t correlation(Int_t row, Int_t col) const;
  Double_t covariance(Int_t row, Int_t col) const;

  Int_t    _status ;          // MINUIT status code
  Int_t    _covQual ;         // MUINUIT quality code of covariance matrix
  Int_t    _numBadNLL ;       // Number calls with bad (zero,negative) likelihood 
  Double_t _minNLL ;          // NLL at minimum
  Double_t _edm ;             // Estimated distance to minimum
  RooArgList* _constPars ;    // List of constant parameters
  RooArgList* _initPars ;     // List of floating parameters with initial values
  RooArgList* _finalPars ;    // List of floating parameters with final values
  RooArgList* _globalCorr ;   // List of global correlation coefficients
  TList       _corrMatrix ;   // Correlation matrix (list of RooArgLists)

  mutable RooArgList *_randomPars; //! List of floating parameters with most recent random perturbation applied
  mutable TMatrixF* _Lt;            //! triangular matrix used for generate random perturbations

  ClassDef(RooFitResult,1) // Container class for fit result
};

#endif
