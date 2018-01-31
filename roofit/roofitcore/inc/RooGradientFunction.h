/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   VC, Vince Croft,     DIANA / NYU,        vincent.croft@cern.ch          *
 *   AL, Alfio Lazzaro,   INFN Milan,         alfio.lazzaro@mi.infn.it       *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_GRADIENT_FUNCTION
#define ROO_GRADIENT_FUNCTION

#include <iostream>
#include <fstream>
#include <vector>

#include "Math/IFunction.h"  // ROOT::Math::IMultiGradFunction
#include "Fit/ParameterSettings.h"
#include "NumericalDerivatorMinuit2.h"

#include "TMatrixDSym.h"

class RooAbsReal;

#include "RooArgList.h"

#include "Minuit2/FunctionGradient.h"

class RooGradientFunction : public ROOT::Math::IMultiGradFunction {
  // An internal implementation class of all the function parts of
  // IMultiGradFunction, to which we will pass on all overrides from
  // RooGradientFunction. This is necessary so we can pass the fully
  // constructed Function to the Derivator. Otherwise, you'll have to
  // either pass the not yet fully constructed this-pointer, or do
  // deferred initialization. This way, everything can be handled by
  // the constructor.
  struct Function : public ROOT::Math::IMultiGenFunction {
    // mutables are because ROOT::Math::IMultiGenFunction::DoEval is const
    mutable Int_t _evalCounter = 0;
    RooAbsReal *_funct;

    // Reset the *largest* negative log-likelihood value we have seen so far:
    mutable double _maxFCN = -1e30;
    mutable int _numBadNLL = 0;
    mutable int _printEvalErrors = 10;
    Bool_t _doEvalErrorWall = kTRUE;

    unsigned int _nDim = 0;

    // put _verbose here, since Function also needs it and RooGradientFunction
    // can still reach it here as well
    bool _verbose;

    RooArgList *_floatParamList;
    std::vector<RooAbsArg *> _floatParamVec;
    RooArgList *_constParamList;
    RooArgList *_initFloatParamList;
    RooArgList *_initConstParamList;

    Function(RooAbsReal *funct, bool verbose);
    Function(const Function& other);
    ~Function() override;

    // overrides of ROOT::Math::IMultiGenFunction (pure) virtuals
    ROOT::Math::IMultiGenFunction *Clone() const override;
    unsigned int NDim() const override { return _nDim; }

    void updateFloatVec();

  private:
    double DoEval(const double *x) const override;
  };

public:
  enum class GradientCalculatorMode {
    ExactlyMinuit2, AlmostMinuit2
  };

private:
  // CAUTION: do not move _function below _gradf, as it is needed for _gradf
  //          construction
  Function _function;

  // mutables are because ROOT::Math::IMultiGradFunction::DoDerivative is const

  // CAUTION: do not move _gradf above _function, as it is needed for _gradf
  //          construction
  mutable RooFit::NumericalDerivatorMinuit2 _gradf;
  mutable ROOT::Minuit2::FunctionGradient _grad;
  mutable std::vector<double> _grad_params;
  std::vector<ROOT::Fit::ParameterSettings> _parameter_settings;

  double DoEval(const double *x) const override;
  double DoDerivative(const double *x, unsigned int icoord) const override;

  void run_derivator(const double *x) const;

  bool hasG2ndDerivative() const override;
  double DoSecondDerivative(const double *x, unsigned int icoord) const override;
  bool hasGStepSize() const override;
  double DoStepSize(const double *x, unsigned int icoord) const override;

  Double_t GetPdfParamVal(Int_t index);
  Double_t GetPdfParamErr(Int_t index);
  void SetPdfParamErr(Int_t index, Double_t value);
  void ClearPdfParamAsymErr(Int_t index);
  void SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal);
  inline Bool_t SetPdfParamVal(const Int_t &index, const Double_t &value) const;

  virtual std::vector<ROOT::Fit::ParameterSettings>& parameter_settings() const;

public:
  explicit RooGradientFunction(RooAbsReal *funct,
                               GradientCalculatorMode grad_mode = GradientCalculatorMode::ExactlyMinuit2,
                               bool verbose = false);
  RooGradientFunction(const RooGradientFunction &other);

  ROOT::Math::IMultiGradFunction *Clone() const override;

  Bool_t synchronize_parameter_settings(std::vector<ROOT::Fit::ParameterSettings>& parameter_settings,
                                        Bool_t optConst = kTRUE, Bool_t verbose = kFALSE);
  void synchronize_gradient_parameter_settings(std::vector<ROOT::Fit::ParameterSettings>& parameter_settings) const;

  bool returnsInMinuit2ParameterSpace() const override;

  unsigned int NDim() const override;

  RooArgList *GetFloatParamList();
  RooArgList *GetConstParamList();
  RooArgList *GetInitFloatParamList();
  RooArgList *GetInitConstParamList();

  void SetEvalErrorWall(Bool_t flag);
  void SetPrintEvalErrors(Int_t numEvalErrors);

  Double_t &GetMaxFCN();
  Int_t GetNumInvalidNLL();

  Int_t evalCounter() const;
  void zeroEvalCount();

  void SetVerbose(Bool_t flag = kTRUE);
};

#endif //ROO_GRADIENT_FUNCTION
