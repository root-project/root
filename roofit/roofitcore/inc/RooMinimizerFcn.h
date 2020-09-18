/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       * 
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef __ROOFIT_NOROOMINIMIZER

#ifndef ROO_MINIMIZER_FCN
#define ROO_MINIMIZER_FCN

#include "Math/IFunction.h"
#include "Fit/ParameterSettings.h"
#include "Fit/FitResult.h"

#include "RooAbsReal.h"
#include "RooArgList.h"

#include <fstream>
#include <vector>

class RooMinimizer;
template<typename T> class TMatrixTSym;
using TMatrixDSym = TMatrixTSym<double>;

class RooMinimizerFcn : public ROOT::Math::IBaseFunctionMultiDim {

 public:

  RooMinimizerFcn(RooAbsReal *funct, RooMinimizer *context, 
	       bool verbose = false);
  RooMinimizerFcn(const RooMinimizerFcn& other);
  virtual ~RooMinimizerFcn();

  virtual ROOT::Math::IBaseFunctionMultiDim* Clone() const;
  virtual unsigned int NDim() const { return _nDim; }

  RooArgList* GetFloatParamList() { return _floatParamList; }
  RooArgList* GetConstParamList() { return _constParamList; }
  RooArgList* GetInitFloatParamList() { return _initFloatParamList; }
  RooArgList* GetInitConstParamList() { return _initConstParamList; }

  void SetEvalErrorWall(Bool_t flag) { _doEvalErrorWall = flag ; }
  void SetPrintEvalErrors(Int_t numEvalErrors) { _printEvalErrors = numEvalErrors ; }
  Bool_t SetLogFile(const char* inLogfile);
  std::ofstream* GetLogFile() { return _logfile; }
  void SetVerbose(Bool_t flag=kTRUE) { _verbose = flag ; }

  Double_t& GetMaxFCN() { return _maxFCN; }
  Int_t GetNumInvalidNLL() const { return _numBadNLL; }

  Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings>& parameters, 
		     Bool_t optConst, Bool_t verbose);
  void BackProp(const ROOT::Fit::FitResult &results);  
  void ApplyCovarianceMatrix(TMatrixDSym& V); 

  Int_t evalCounter() const { return _evalCounter ; }
  void zeroEvalCount() { _evalCounter = 0 ; }


 private:
  void SetPdfParamErr(Int_t index, Double_t value);
  void ClearPdfParamAsymErr(Int_t index);
  void SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal);

  Bool_t SetPdfParamVal(int index, double value) const;
  void printEvalErrors() const;

  virtual double DoEval(const double * x) const;  


  RooAbsReal *_funct;
  const RooMinimizer *_context;

  mutable double _maxFCN;
  mutable int _numBadNLL;
  mutable int _printEvalErrors;
  mutable int _evalCounter{0};
  int _nDim;

  RooArgList* _floatParamList;
  RooArgList* _constParamList;
  RooArgList* _initFloatParamList;
  RooArgList* _initConstParamList;

  std::ofstream *_logfile;
  bool _doEvalErrorWall;
  bool _verbose;

};

#endif
#endif
