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


#include "Math/IFunction.h" // ROOT::Math::IMultiGradFunction
#include "Fit/ParameterSettings.h"
#include "Fit/FitResult.h"
#include "NumericalDerivatorMinuit2.h"

#include "TMatrixDSym.h"

//#include "RooAbsReal.h"
class RooAbsReal;
#include "RooArgList.h"

#include <iostream>
#include <fstream>

#include "Minuit2/FunctionGradient.h"

class RooGradientFunction : public ROOT::Math::IMultiGradFunction {

public:

  enum class GradientCalculatorMode {ExactlyMinuit2, AlmostMinuit2};

  RooGradientFunction(RooAbsReal *funct,
                      GradientCalculatorMode grad_mode = ExactlyMinuit2);

  RooGradientFunction(const RooGradientFunction &other);

  virtual ~RooGradientFunction();

  ROOT::Math::IMultiGradFunction *Clone() const override;

  unsigned int NDim() const override { return _nDim; }

  RooArgList *GetFloatParamList() { return _floatParamList; }

  RooArgList *GetConstParamList() { return _constParamList; }

  RooArgList *GetInitFloatParamList() { return _initFloatParamList; }

  RooArgList *GetInitConstParamList() { return _initConstParamList; }

  void SetEvalErrorWall(Bool_t flag) { _doEvalErrorWall = flag; }

  void SetPrintEvalErrors(Int_t numEvalErrors) { _printEvalErrors = numEvalErrors; }

  Bool_t SetLogFile(const char *inLogfile);

  std::ofstream *GetLogFile() { return _logfile; }

  void SetVerbose(Bool_t flag = kTRUE);

  Double_t &GetMaxFCN() { return _maxFCN; }

  Int_t GetNumInvalidNLL() { return _numBadNLL; }

  Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters,
                     Bool_t optConst, Bool_t verbose);

  void BackProp(const ROOT::Fit::FitResult &results);

  void ApplyCovarianceMatrix(TMatrixDSym &V);

  Int_t evalCounter() const { return _evalCounter; }

  void zeroEvalCount() { _evalCounter = 0; }

  void SynchronizeGradient(std::vector<ROOT::Fit::ParameterSettings> &parameters) const;

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

  mutable Int_t _evalCounter;

  RooAbsReal *_funct;
  RooGradMinimizer *_context;

  mutable double _maxFCN;
  mutable int _numBadNLL;
  mutable int _printEvalErrors;
  Bool_t _doEvalErrorWall;

  int _nDim;
  std::ofstream *_logfile;
  bool _verbose;

  RooArgList *_floatParamList;
  std::vector<RooAbsArg *> _floatParamVec;
  RooArgList *_constParamList;
  RooArgList *_initFloatParamList;
  RooArgList *_initConstParamList;

  // Before using any of the following members, call InitGradient!
  // this all needs to be mutable since ROOT::Math::IMultiGradFunction insists on DoDerivative being const
  mutable RooFit::NumericalDerivatorMinuit2 _gradf;
//  mutable std::vector<double> _grad;
  mutable ROOT::Minuit2::FunctionGradient _grad;
  mutable std::vector<double> _grad_params;
  mutable bool _grad_initialized;

  void InitGradient() const;

  bool hasG2ndDerivative() const override;
  bool hasGStepSize() const override;
  double DoSecondDerivative(const double *x, unsigned int icoord) const override;
  double DoStepSize(const double *x, unsigned int icoord) const override;

private:
  bool _always_exactly_mimic_minuit2; //! only used for initializing the derivator
public:
  bool always_exactly_mimic_minuit2() const;
  bool set_always_exactly_mimic_minuit2(bool flag = true) const;

};
#endif //ROO_GRADIENT_FUNCTION
