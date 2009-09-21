/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooFitResult.h,v 1.28 2007/05/11 09:11:30 verkerke Exp $
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

#include "Riosfwd.h"
#include "RooAbsArg.h"
#include "RooPrintable.h"
#include "RooDirItem.h"
#include "RooArgList.h"

#include "RVersion.h"
#include "TMatrixFfwd.h"
#include "TMatrixDSym.h"
#include "TRootIOCtor.h"

class RooArgSet ;
class RooAbsPdf ;
class RooPlot;
class TObject ;
class TH2 ;
typedef RooArgSet* pRooArgSet ;

class RooFitResult : public TNamed, public RooPrintable, public RooDirItem {
public:
 
  // Constructors, assignment etc.
  RooFitResult(const char* name=0, const char* title=0) ;
  RooFitResult(const RooFitResult& other) ;   			     
  virtual TObject* Clone(const char* newname = 0) const { 
    RooFitResult* r =  new RooFitResult(*this) ; 
    if (newname && *newname) r->SetName(newname) ; 
    return r ; 
  }
  virtual TObject* clone() const { return new RooFitResult(*this); }   
  virtual ~RooFitResult() ;

  static RooFitResult* lastMinuitFit(const RooArgList& varList=RooArgList()) ;

  // Printing interface (human readable)
  virtual void printValue(ostream& os) const ;
  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;
  void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  inline virtual void Print(Option_t *options= 0) const {
    // Printing interface
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual Int_t defaultPrintContents(Option_t* opt) const ;
  virtual StyleOption defaultPrintStyle(Option_t* opt) const ;

  RooAbsPdf* createHessePdf(const RooArgSet& params) const ;

  // Accessors
  inline Int_t status() const {
    // Return MINUIT status code
    return _status ; 
  }
  inline Int_t covQual() const { 
    // Return MINUIT quality code of covariance matrix
    return _covQual ; 
  }
  inline Int_t numInvalidNLL() const { 
    // Return number of NLL evaluations with problems
    return _numBadNLL ; 
  }
  inline Double_t edm() const { 
    // Return estimated distance to minimum
    return _edm ; 
  }
  inline Double_t minNll() const { 
    // Return minimized -log(L) value
    return _minNLL ; 
  }
  inline const RooArgList& constPars() const { 
    // Return list of constant parameters
    return *_constPars ; 
  }
  inline const RooArgList& floatParsInit() const { 
    // Return list of floating parameters before fit
    return *_initPars ; 
  } 
  inline const RooArgList& floatParsFinal() const { 
    // Return list of floarting parameters after fit
    return *_finalPars ; 
  } 

  TH2* correlationHist(const char* name = "correlation_matrix") const ;

  Double_t correlation(const RooAbsArg& par1, const RooAbsArg& par2) const {
    // Return correlation between par1 and par2
    return correlation(par1.GetName(),par2.GetName()) ;
  }
  const RooArgList* correlation(const RooAbsArg& par) const {
    // Return pointer to list of correlations of all parameters with par
    return correlation(par.GetName()) ;
  }

  Double_t correlation(const char* parname1, const char* parname2) const ;
  const RooArgList* correlation(const char* parname) const ;

  
  const TMatrixDSym& covarianceMatrix() const ;
  TMatrixDSym reducedCovarianceMatrix(const RooArgList& params) const ;
  const TMatrixDSym& correlationMatrix() const ;


  // Global correlation accessors
  Double_t globalCorr(const RooAbsArg& par) { return globalCorr(par.GetName()) ; }
  Double_t globalCorr(const char* parname) ;
  const RooArgList* globalCorr() ;


  // Add objects to a 2D plot
  inline RooPlot *plotOn(RooPlot *frame, const RooAbsArg &par1, const RooAbsArg &par2,
			 const char *options= "ME") const {
    // Plot error ellipse in par1 and par2 on frame
    return plotOn(frame,par1.GetName(),par2.GetName(),options);
  }
  RooPlot *plotOn(RooPlot *plot, const char *parName1, const char *parName2,
		  const char *options= "ME") const;

  // Generate random perturbations of the final parameters using the covariance matrix
  const RooArgList& randomizePars() const;

  Bool_t isIdentical(const RooFitResult& other, Double_t tol=5e-5, Double_t tolCorr=1e-4, Bool_t verbose=kTRUE) const ;

  void SetName(const char *name) ;
  void SetNameTitle(const char *name, const char* title) ;

protected:
  
  friend class RooMinuit ;
  friend class RooMinimizer ;
  friend class RooNag ;
  void setCovarianceMatrix(TMatrixDSym& V) ; 
  void setConstParList(const RooArgList& list) ;
  void setInitParList(const RooArgList& list) ;
  void setFinalParList(const RooArgList& list) ;
  inline void setMinNLL(Double_t val) { _minNLL = val ; }
  inline void setEDM(Double_t val) { _edm = val ; }
  inline void setStatus(Int_t val) { _status = val ; }
  inline void setCovQual(Int_t val) { _covQual = val ; }
  inline void setNumInvalidNLL(Int_t val) { _numBadNLL=val ; }
  void fillCorrMatrix() ;
  void fillCorrMatrix(const std::vector<double>& globalCC, const TMatrixDSym& corrs, const TMatrixDSym& covs) ;
  void fillLegacyCorrMatrix() const ;

  Double_t correlation(Int_t row, Int_t col) const;
  Double_t covariance(Int_t row, Int_t col) const;

  Int_t    _status ;          // MINUIT status code
  Int_t    _covQual ;         // MINUIT quality code of covariance matrix
  Int_t    _numBadNLL ;       // Number calls with bad (zero,negative) likelihood 
  Double_t _minNLL ;          // NLL at minimum
  Double_t _edm ;             // Estimated distance to minimum
  RooArgList* _constPars ;    // List of constant parameters
  RooArgList* _initPars ;     // List of floating parameters with initial values
  RooArgList* _finalPars ;    // List of floating parameters with final values

  mutable RooArgList* _globalCorr ;   //! List of global correlation coefficients
  mutable TList       _corrMatrix ;   //! Correlation matrix (list of RooArgLists)

  mutable RooArgList *_randomPars; //! List of floating parameters with most recent random perturbation applied
  mutable TMatrixF* _Lt;            //! triangular matrix used for generate random perturbations

  TMatrixDSym* _CM ;  // Correlation matrix 
  TMatrixDSym* _VM ;  // Covariance matrix 
  TVectorD* _GC ;     // Global correlation coefficients 

  ClassDef(RooFitResult,4) // Container class for fit result
};

#endif
