/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara,   verkerke@slac.stanford.edu     *
 *   DK, David Kirkby,    UC Irvine,          dkirkby@uci.edu                *
 *   AL, Alfio Lazzaro,   INFN Milan,         alfio.lazzaro@mi.infn.it       *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef __ROOFIT_NOROOMINIMIZER

#ifndef ROO_MINIMIZER
#define ROO_MINIMIZER

#include "TObject.h"
#include "TStopwatch.h"
#include <fstream>
#include "TMatrixDSymfwd.h"

#include "RooArgList.h" // cannot just use forward decl due to default argument in lastMinuitFit

#include "Fit/Fitter.h"
#include "RooMinimizerType.h"
//#include <ROOT/RMakeUnique.hxx>  // make_unique
#include <memory>  // make_shared

class RooAbsReal ;
class RooFitResult ;
class RooRealVar ;
class RooArgSet ;
class TH2F ;
class RooPlot ;


template <class MinimizerFcn, RooFit::MinimizerType default_minimizer_type,
    typename... MinimizerFcnCtorAdditionalArgs>
class RooMinimizerTemplate : public TObject {
public:

  RooMinimizerTemplate(RooAbsReal& function, MinimizerFcnCtorAdditionalArgs... minFcnCArgs) ;
  virtual ~RooMinimizerTemplate() ;

  enum Strategy { Speed=0, Balance=1, Robustness=2 } ;
  enum PrintLevel { None=-1, Reduced=0, Normal=1, ExtraForProblem=2, Maximum=3 } ;
  void setStrategy(Int_t strat) ;
  void setErrorLevel(Double_t level) ;
  void setEps(Double_t eps) ;
  void optimizeConst(Int_t flag) ;
  void setEvalErrorWall(Bool_t flag) { fitterFcn()->SetEvalErrorWall(flag); }
  void setOffsetting(Bool_t flag) ;
  void setMaxIterations(Int_t n) ;
  void setMaxFunctionCalls(Int_t n) ; 

  RooFitResult* fit(const char* options) ;

  Int_t migrad() ;
  Int_t hesse() ;
  Int_t minos() ;
  Int_t minos(const RooArgSet& minosParamList) ;
  Int_t seek() ;
  Int_t simplex() ;
  Int_t improve() ;

  Int_t minimize(const char* type, const char* alg=0) ;

  RooFitResult* save(const char* name=0, const char* title=0) ;
  RooPlot* contour(RooRealVar& var1, RooRealVar& var2, 
                   Double_t n1=1, Double_t n2=2, Double_t n3=0,
                   Double_t n4=0, Double_t n5=0, Double_t n6=0) ;

  Int_t setPrintLevel(Int_t newLevel) ; 
  void setPrintEvalErrors(Int_t numEvalErrors) { fitterFcn()->SetPrintEvalErrors(numEvalErrors); }
  void setVerbose(Bool_t flag=kTRUE) { _verbose = flag ; fitterFcn()->SetVerbose(flag); }
  void setProfile(Bool_t flag=kTRUE) { _profile = flag ; }
  Bool_t setLogFile(const char* logf=0) { return fitterFcn()->SetLogFile(logf); }

  void setMinimizerType(const char* type) ;

  static void cleanup() ;
  static RooFitResult* lastMinuitFit(const RooArgList& varList=RooArgList()) ;

  void saveStatus(const char* label, Int_t status) { _statusHistory.push_back(std::pair<std::string,int>(label,status)) ; }

  Int_t evalCounter() const { return fitterFcn()->evalCounter() ; }
  void zeroEvalCount() { fitterFcn()->zeroEvalCount() ; }

  ROOT::Fit::Fitter* fitter() ;
  const ROOT::Fit::Fitter* fitter() const ;
  
protected:

  friend class RooAbsPdf ;
  void applyCovarianceMatrix(TMatrixDSym& V) ;

  void profileStart() ;
  void profileStop() ;

  inline Int_t getNPar() const { return fitterFcn()->NDim() ; }
  inline std::ofstream* logfile() { return fitterFcn()->GetLogFile(); }
  inline Double_t& maxFCN() { return fitterFcn()->GetMaxFCN() ; }

  // dynamic cast necessary due to e.g. GradMinimizerFcn which has virtual inheritance
  const MinimizerFcn* fitterFcn() const {  return ( fitter()->GetFCN() ? (dynamic_cast<MinimizerFcn*>(fitter()->GetFCN())) : _fcn ) ; }
  MinimizerFcn* fitterFcn() { return ( fitter()->GetFCN() ? (dynamic_cast<MinimizerFcn*>(fitter()->GetFCN())) : _fcn ) ; }

private:

  Int_t       _printLevel = 1;
  Int_t       _status ;
  Bool_t      _optConst = kFALSE;
  Bool_t      _profile = kFALSE;
  RooAbsReal* _func ;

  Bool_t      _verbose = kFALSE;
  TStopwatch  _timer ;
  TStopwatch  _cumulTimer ;
  Bool_t      _profileStart = kFALSE;

  TMatrixDSym* _extV = 0;

  MinimizerFcn *_fcn;
  std::string _minimizerType = RooFit::minimizer_type(default_minimizer_type);

  static ROOT::Fit::Fitter *_theFitter ;

  std::vector<std::pair<std::string,int> > _statusHistory ;

  RooMinimizerTemplate(const RooMinimizerTemplate&) ;

  ClassDef(RooMinimizerTemplate,0) // RooFit interface to ROOT::Fit::Fitter
};


// In RooGradMinimizerFcn (and maybe other places) we need to erase the exact
// type of the RooMinimizerTemplate instance, so that we can pass any instance
// type as an argument, without needing those instances to inherit from a
// common base class.

class RooMinimizerGenericPtr {
  struct InnerBase {
    virtual ~InnerBase() = default; // virtual dtor necessary for silencing warnings from -Wdelete-non-virtual-dtor; the destruction of val below (shared_ptr<InnerBase>) otherwise will not call the destructor of Inner, because it only knows about the non-virtual InnerBase dtor
    virtual ROOT::Fit::Fitter* fitter() const = 0;
    virtual TObject * get_ptr() const = 0;
  };

  template<typename T>
  struct Inner : public InnerBase {
    Inner(T * const ptr) {
      _ptr = ptr;
    };
    ROOT::Fit::Fitter* fitter() const override {
      return _ptr->fitter();
    };
    // note that this constrains T to be a TObject subclass:
    TObject * get_ptr() const override {
      return static_cast<TObject *>(_ptr);
    }

   private:
    T* _ptr;
  };

  std::shared_ptr<InnerBase> val;

 public:
  template <typename T>
  explicit RooMinimizerGenericPtr(T * const roo_minimizer_tmpl_inst) :
      val(std::make_shared< RooMinimizerGenericPtr::Inner<T> >(roo_minimizer_tmpl_inst)) {};

  // fitter() is needed in RooGradMinimizerFcn::parameter_settings:
  ROOT::Fit::Fitter* fitter() const;
  // the logging functions in RooMsgService need to get a raw pointer
  operator TObject * () const;
};

#endif

#endif

// template implementation file
#include "RooMinimizer_impl.h"


// template instantiation for backwards compatibility
#include <RooMinimizerFcn.h>
// we must also forward declare RooMinimizerFcn, because when some file only
// includes RooMinimizerFcn.h, it will first include RooMinimizer.h from
// RooMinimizerFcn.h before actually defining RooMinimizerFcn, so then at
// this point the above RooMinimizerFcn.h include won't include the
// RooMinimizerFcn definition in this file, because the include guard will
// have been defined by the initial RooMinimizerFcn.h include
class RooMinimizerFcn;
using RooMinimizer = RooMinimizerTemplate<RooMinimizerFcn, RooFit::MinimizerType::Minuit>;
