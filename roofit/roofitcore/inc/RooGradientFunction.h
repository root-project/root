/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
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
#include "Fit/FitResult.h"
#include "NumericalDerivatorMinuit2.h"

#include "TMatrixDSym.h"

class RooAbsReal;

#include "RooArgList.h"

#include "Minuit2/FunctionGradient.h"

class RooGradientFunction : public ROOT::Math::IMultiGradFunction {
private:
  // an internal implementation class of all the function parts of IMultiGradFunction,
  // to which we will pass on all overrides from RooGradientFunction
  struct Function : public ROOT::Math::IMultiGenFunction {
    mutable Int_t _evalCounter = 0;
    RooAbsReal *_funct;

    mutable double _maxFCN = -1e30;  // Reset the *largest* negative log-likelihood value we have seen so far
    mutable int _numBadNLL = 0;
    mutable int _printEvalErrors = 10;
    Bool_t _doEvalErrorWall = kTRUE;

    unsigned int _nDim = 0;

    RooArgList *_floatParamList;
    std::vector<RooAbsArg *> _floatParamVec;
    RooArgList *_constParamList;
    RooArgList *_initFloatParamList;
    RooArgList *_initConstParamList;

    Function(RooAbsReal *funct);
    Function(const Function& other);
    ~Function() override;


  };

public:
  enum class GradientCalculatorMode {
    ExactlyMinuit2, AlmostMinuit2
  };

  explicit RooGradientFunction(RooAbsReal *funct,
                               GradientCalculatorMode grad_mode = GradientCalculatorMode::ExactlyMinuit2);

  RooGradientFunction(const RooGradientFunction &other);

  ~RooGradientFunction() override;

  ROOT::Math::IMultiGradFunction *Clone() const override;

  unsigned int NDim() const override { return _function._nDim; }

  RooArgList *GetFloatParamList() { return _function._floatParamList; }

  RooArgList *GetConstParamList() { return _function._constParamList; }

  RooArgList *GetInitFloatParamList() { return _function._initFloatParamList; }

  RooArgList *GetInitConstParamList() { return _function._initConstParamList; }

  void SetEvalErrorWall(Bool_t flag) { _function._doEvalErrorWall = flag; }

  void SetPrintEvalErrors(Int_t numEvalErrors) { _function._printEvalErrors = numEvalErrors; }

  Double_t &GetMaxFCN() { return _function._maxFCN; }

  Int_t GetNumInvalidNLL() { return _function._numBadNLL; }

  Bool_t synchronize_parameter_settings(Bool_t optConst);
  void synchronize_gradient_parameter_settings() const;

  void BackProp(const ROOT::Fit::FitResult &results);

  void ApplyCovarianceMatrix(TMatrixDSym &V);

  Int_t evalCounter() const { return _function._evalCounter; }

  void zeroEvalCount() { _function._evalCounter = 0; }


  bool returnsInMinuit2ParameterSpace() const override;

private:

  Double_t GetPdfParamVal(Int_t index);

  Double_t GetPdfParamErr(Int_t index);

  void SetPdfParamErr(Int_t index, Double_t value);

  void ClearPdfParamAsymErr(Int_t index);

  void SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal);

  inline Bool_t SetPdfParamVal(const Int_t &index, const Double_t &value) const;


  double DoEval(const double *x) const override;

  void updateFloatVec();

  double DoDerivative(const double *x, unsigned int icoord) const override;

  void run_derivator(const double *x) const;

private:
  // CAUTION: do not move _function below _gradf, as it is needed for _gradf construction
  Function _function;

private:
  GradientCalculatorMode _grad_mode; //! only used for initializing the derivator
public:
  GradientCalculatorMode grad_mode() const;

private:
  // Before using any of the following members, call InitGradient!
  // this all needs to be mutable since ROOT::Math::IMultiGradFunction insists on DoDerivative being const

  // CAUTION: do not move _gradf above _function, as it is needed for _gradf construction
  mutable RooFit::NumericalDerivatorMinuit2 _gradf {_function,
                                                    _grad_mode == GradientCalculatorMode::ExactlyMinuit2 ? true : false};
//  mutable std::vector<double> _grad;
  mutable ROOT::Minuit2::FunctionGradient _grad;
  mutable std::vector<double> _grad_params;
  mutable bool _grad_initialized;

  void InitGradient() const;

  bool hasG2ndDerivative() const override;

  bool hasGStepSize() const override;

  double DoSecondDerivative(const double *x, unsigned int icoord) const override;

  double DoStepSize(const double *x, unsigned int icoord) const override;

  // necessary for NumericalDerivatorMinuit2
  std::vector<ROOT::Fit::ParameterSettings> parameter_settings;


};

#endif //ROO_GRADIENT_FUNCTION
