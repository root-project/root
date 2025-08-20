/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealIntegral.h,v 1.44 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_INTEGRAL
#define ROO_REAL_INTEGRAL

#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooListProxy.h>
#include <RooRealProxy.h>
#include <RooSetProxy.h>

class RooAbsIntegrator;
class RooNumIntConfig;

class RooRealIntegral : public RooAbsReal {
public:

  // Constructors, assignment etc
  RooRealIntegral() ;
  RooRealIntegral(const char *name, const char *title, const RooAbsReal& function, const RooArgSet& depList,
        const RooArgSet* funcNormSet=nullptr, const RooNumIntConfig* config=nullptr, const char* rangeName=nullptr) ;
  RooRealIntegral(const RooRealIntegral& other, const char* name=nullptr);
  TObject* clone(const char* newname=nullptr) const override { return new RooRealIntegral(*this,newname); }
  ~RooRealIntegral() override;

  double getValV(const RooArgSet* set=nullptr) const override ;

  bool isValid() const override { return _valid; }

  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;
  void printMetaArgs(std::ostream& os) const override ;

  const RooArgSet& numIntCatVars() const { return _sumList ; }
  const RooArgSet& numIntRealVars() const { return _intList ; }
  const RooArgSet& anaIntVars() const { return _anaList ; }

  RooArgSet intVars() const { RooArgSet tmp(_sumList) ; tmp.add(_intList) ; tmp.add(_anaList) ; tmp.add(_facList) ; return tmp ; }
  const char* intRange() const { return _rangeName ? _rangeName->GetName() : nullptr ; }
  const RooAbsReal& integrand() const { return *_function; }

  void setCacheNumeric(bool flag) {
    // If true, value of this integral is cached if it is (partially numeric)
    _cacheNum = flag ;
  }

  bool getCacheNumeric() {
    // If true, value of this integral is cached if it is (partially numeric)
    return _cacheNum ;
  }

  static void setCacheAllNumeric(Int_t ndim) ;

  static Int_t getCacheAllNumeric() ;

  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override {
    // Forward plot sampling hint of integrand
    return _function->plotSamplingHint(obs,xlo,xhi) ;
  }

  RooFit::OwningPtr<RooAbsReal> createIntegral(const RooArgSet& iset, const RooArgSet* nset=nullptr, const RooNumIntConfig* cfg=nullptr, const char* rangeName=nullptr) const override ;

  void setAllowComponentSelection(bool allow);
  bool getAllowComponentSelection() const;

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

  inline RooArgSet const* funcNormSet() const { return _funcNormSet.get(); }

  int mode() const { return _mode; }

protected:

  mutable bool _valid = false;
  bool _respectCompSelect = true;

  const RooArgSet& parameters() const ;

  enum IntOperMode { Hybrid, Analytic, PassThrough } ;
  //friend class RooAbsPdf ;

  bool initNumIntegrator() const;
  void autoSelectDirtyMode() ;

  virtual double sum() const ;
  virtual double integrate() const ;
  virtual double jacobianProduct() const ;

  // Evaluation and validation implementation
  double evaluate() const override ;
  bool isValidReal(double value, bool printError=false) const override ;

  bool redirectServersHook(const RooAbsCollection& newServerList,
                 bool mustReplaceAll, bool nameChange, bool isRecursive) override ;

  // Internal function to get the normalization set for the integrated
  // function. By default, we will take the normalization set from the function
  // proxy, but _funcNormSet will be used if it is set.
  inline RooArgSet const* actualFuncNormSet() const {
    return _funcNormSet ? _funcNormSet.get() : _function.nset();
  }

  // Function pointer and integrands list
  RooSetProxy _sumList; ///< Set of discrete observable over which is summed numerically
  RooSetProxy _intList; ///< Set of continuous observables over which is integrated numerically
  RooSetProxy _anaList; ///< Set of observables over which is integrated/summed analytically
  RooSetProxy _jacList; ///< Set of lvalue observables over which is analytically integration that have a non-unit Jacobian
  RooSetProxy _facList; ///< Set of observables on which function does not depends, which are integrated nevertheless

  RooRealProxy       _function ;     ///< Function being integrated
  std::unique_ptr<RooArgSet> _funcNormSet; ///< Optional normalization set passed to function

  RooArgSet _saveInt; ///<!
  RooArgSet _saveSum; ///<!

  RooNumIntConfig* _iconfig = nullptr;

  RooListProxy _sumCat ; ///<!

  Int_t _mode = 0;
  IntOperMode _intOperMode = Hybrid;   ///< integration operation mode

  mutable bool _restartNumIntEngine = false; ///<!
  mutable std::unique_ptr<RooAbsIntegrator> _numIntEngine;  ///<!
  mutable std::unique_ptr<RooAbsFunc> _numIntegrand;        ///<!

  TNamed* _rangeName = nullptr;

  mutable std::unique_ptr<RooArgSet> _params; ///<! cache for set of parameters

  bool _cacheNum = false;           ///< Cache integral if numeric
  static Int_t _cacheAllNDim ; ///<! Cache all integrals with given numeric dimension

private:
  void addNumIntDep(RooAbsArg const &arg);

  ClassDefOverride(RooRealIntegral,5) // Real-valued function representing an integral over a RooAbsReal object
};

#endif
