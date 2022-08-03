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

#include "RooAbsArg.h"
#include "RooPrintable.h"
#include "RooDirItem.h"
#include "RooArgList.h"

#include "RVersion.h"
#include "TMatrixFfwd.h"
#include "TMatrixDSym.h"
#include "TList.h"

#include <vector>
#include <string>
#include <utility>

class RooArgSet ;
class RooAbsPdf ;
class RooPlot;
class TObject ;
class TH2 ;
typedef RooArgSet* pRooArgSet ;

class RooFitResult : public TNamed, public RooPrintable, public RooDirItem {
public:

  // Constructors, assignment etc.
  RooFitResult(const char* name=nullptr, const char* title=nullptr) ;
  RooFitResult(const RooFitResult& other) ;
  TObject* Clone(const char* newname = 0) const override {
    RooFitResult* r =  new RooFitResult(*this) ;
    if (newname && *newname) r->SetName(newname) ;
    return r ;
  }
  virtual TObject* clone() const { return new RooFitResult(*this); }
  ~RooFitResult() override ;

  static RooFitResult* lastMinuitFit(const RooArgList& varList=RooArgList()) ;

  static RooFitResult *prefitResult(const RooArgList &paramList);

  // Printing interface (human readable)
  void printValue(std::ostream& os) const override ;
  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printArgs(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;

  inline void Print(Option_t *options= 0) const override {
    // Printing interface
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  Int_t defaultPrintContents(Option_t* opt) const override ;
  StyleOption defaultPrintStyle(Option_t* opt) const override ;

  RooAbsPdf* createHessePdf(const RooArgSet& params) const ;

  // Accessors
  /// Return MINUIT status code
  inline Int_t status() const {
    return _status ;
  }

  inline UInt_t numStatusHistory() const { return _statusHistory.size() ; }
  Int_t statusCodeHistory(UInt_t icycle) const ;
  const char* statusLabelHistory(UInt_t icycle) const ;

  /// Return MINUIT quality code of covariance matrix
  inline Int_t covQual() const {
    return _covQual ;
  }
  /// Return number of NLL evaluations with problems.
  inline Int_t numInvalidNLL() const {
    return _numBadNLL ;
  }
  /// Return estimated distance to minimum.
  inline double edm() const {
    return _edm ;
  }
  /// Return minimized -log(L) value.
  inline double minNll() const {
    return _minNLL ;
  }
  /// Return list of constant parameters.
  inline const RooArgList& constPars() const {
    return *_constPars ;
  }
  /// Return list of floating parameters before fit.
  inline const RooArgList& floatParsInit() const {
    return *_initPars ;
  }
  /// Return list of floating parameters after fit.
  inline const RooArgList& floatParsFinal() const {
    return *_finalPars ;
  }

  TH2* correlationHist(const char* name = "correlation_matrix") const ;

  /// Return correlation between par1 and par2.
  double correlation(const RooAbsArg& par1, const RooAbsArg& par2) const {
    return correlation(par1.GetName(),par2.GetName()) ;
  }
  /// Return pointer to list of correlations of all parameters with par.
  const RooArgList* correlation(const RooAbsArg& par) const {
    return correlation(par.GetName()) ;
  }

  double correlation(const char* parname1, const char* parname2) const ;
  const RooArgList* correlation(const char* parname) const ;


  const TMatrixDSym& covarianceMatrix() const ;
  const TMatrixDSym& correlationMatrix() const ;
  TMatrixDSym reducedCovarianceMatrix(const RooArgList& params) const ;
  TMatrixDSym conditionalCovarianceMatrix(const RooArgList& params) const ;


  // Global correlation accessors
  double globalCorr(const RooAbsArg& par) { return globalCorr(par.GetName()) ; }
  double globalCorr(const char* parname) ;
  const RooArgList* globalCorr() ;


  /// Add objects to a 2D plot.
  /// Plot error ellipse in par1 and par2 on frame.
  inline RooPlot *plotOn(RooPlot *frame, const RooAbsArg &par1, const RooAbsArg &par2,
          const char *options= "ME") const {
    return plotOn(frame,par1.GetName(),par2.GetName(),options);
  }
  RooPlot *plotOn(RooPlot *plot, const char *parName1, const char *parName2,
        const char *options= "ME") const;

  /// Generate random perturbations of the final parameters using the covariance matrix.
  const RooArgList& randomizePars() const;

  bool isIdenticalNoCov(const RooFitResult& other, double tol=1e-6, double tolErr=1e-3, bool verbose=true) const ;
  bool isIdentical(const RooFitResult& other, double tol=1e-6, double tolCorr=1e-4, bool verbose=true) const ;

  void SetName(const char *name) override ;
  void SetNameTitle(const char *name, const char* title) override ;

protected:

  friend class RooAbsPdf ;
  friend class RooMinuit ;
  friend class RooMinimizer ;
  void setCovarianceMatrix(TMatrixDSym& V) ;
  void setConstParList(const RooArgList& list) ;
  void setInitParList(const RooArgList& list) ;
  void setFinalParList(const RooArgList& list) ;
  inline void setMinNLL(double val) { _minNLL = val ; }
  inline void setEDM(double val) { _edm = val ; }
  inline void setStatus(Int_t val) { _status = val ; }
  inline void setCovQual(Int_t val) { _covQual = val ; }
  inline void setNumInvalidNLL(Int_t val) { _numBadNLL=val ; }
  void fillCorrMatrix() ;
  void fillCorrMatrix(const std::vector<double>& globalCC, const TMatrixDSym& corrs, const TMatrixDSym& covs) ;
  void fillLegacyCorrMatrix() const ;
  void fillPrefitCorrMatrix();
  void setStatusHistory(std::vector<std::pair<std::string,int> >& hist) { _statusHistory = hist ; }

  double correlation(Int_t row, Int_t col) const;
  double covariance(Int_t row, Int_t col) const;

  Int_t    _status ;          ///< MINUIT status code
  Int_t    _covQual ;         ///< MINUIT quality code of covariance matrix
  Int_t    _numBadNLL ;       ///< Number calls with bad (zero,negative) likelihood
  double _minNLL ;          ///< NLL at minimum
  double _edm ;             ///< Estimated distance to minimum
  RooArgList* _constPars = nullptr;    ///< List of constant parameters
  RooArgList* _initPars = nullptr;     ///< List of floating parameters with initial values
  RooArgList* _finalPars = nullptr;    ///< List of floating parameters with final values

  mutable RooArgList* _globalCorr = nullptr;   ///<! List of global correlation coefficients
  mutable TList       _corrMatrix ;   ///<! Correlation matrix (list of RooArgLists)

  mutable RooArgList *_randomPars = nullptr; ///<! List of floating parameters with most recent random perturbation applied
  mutable TMatrixF* _Lt = nullptr;           ///<! triangular matrix used for generate random perturbations

  TMatrixDSym* _CM = nullptr;  ///< Correlation matrix
  TMatrixDSym* _VM = nullptr;  ///< Covariance matrix
  TVectorD* _GC = nullptr;     ///< Global correlation coefficients

  std::vector<std::pair<std::string,int> > _statusHistory ; ///< History of status codes

  ClassDefOverride(RooFitResult,5) // Container class for fit result
};

#endif
