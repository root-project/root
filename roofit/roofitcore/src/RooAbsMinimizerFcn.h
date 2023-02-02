/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl   *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_ABS_MINIMIZER_FCN
#define ROO_ABS_MINIMIZER_FCN

#include "Math/IFunction.h"
#include "Fit/ParameterSettings.h"
#include "Fit/FitResult.h"

#include "TMatrixDSym.h"

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooMinimizer.h"
#include "RooRealVar.h"

#include <Fit/Fitter.h>

#include <iostream>
#include <fstream>
#include <string>
#include <memory> // unique_ptr

class RooAbsMinimizerFcn {

public:
   RooAbsMinimizerFcn(RooArgList paramList, RooMinimizer *context);
   RooAbsMinimizerFcn(const RooAbsMinimizerFcn &other);
   virtual ~RooAbsMinimizerFcn() = default;

   /// Informs Minuit through its parameter_settings vector of RooFit parameter properties.
   bool synchronizeParameterSettings(std::vector<ROOT::Fit::ParameterSettings> &parameters, bool optConst);
   /// Like synchronizeParameterSettings, Synchronize informs Minuit through
   /// its parameter_settings vector of RooFit parameter properties, but
   /// Synchronize can be overridden to e.g. also include gradient strategy
   /// synchronization in subclasses.
   virtual bool Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters);

   RooArgList *GetFloatParamList() { return _floatParamList.get(); }
   RooArgList *GetConstParamList() { return _constParamList.get(); }
   RooArgList *GetInitFloatParamList() { return _initFloatParamList.get(); }
   RooArgList *GetInitConstParamList() { return _initConstParamList.get(); }
   Int_t GetNumInvalidNLL() const { return _numBadNLL; }

   double &GetMaxFCN() { return _maxFCN; }
   Int_t evalCounter() const { return _evalCounter; }
   void zeroEvalCount() { _evalCounter = 0; }
   /// Return a possible offset that's applied to the function to separate invalid function values from valid ones.
   double getOffset() const { return _funcOffset; }

   /// Put Minuit results back into RooFit objects.
   void BackProp(const ROOT::Fit::FitResult &results);

   /// RooMinimizer sometimes needs the name of the minimized function. Implement this in the derived class.
   virtual std::string getFunctionName() const = 0;
   /// RooMinimizer sometimes needs the title of the minimized function. Implement this in the derived class.
   virtual std::string getFunctionTitle() const = 0;

   /// Set different external covariance matrix
   void ApplyCovarianceMatrix(TMatrixDSym &V);

   bool SetLogFile(const char *inLogfile);
   std::ofstream *GetLogFile() { return _logfile; }

   unsigned int getNDim() const { return _nDim; }

   // In the past, the `getNDim` function was called just `NDim`. The function
   // was renamed to match the code convention (lower case for function names),
   // but we have to keep an overload with the old name to not break existing
   // user code.
   inline unsigned int NDim() const { return getNDim(); }

   void setOptimizeConst(Int_t flag);

   std::vector<double> getParameterValues() const;

   bool SetPdfParamVal(int index, double value) const;

   /// Enable or disable offsetting on the function to be minimized, which enhances numerical precision.
   virtual void setOffsetting(bool flag) = 0;
   virtual bool fit(ROOT::Fit::Fitter &) const = 0;
   virtual ROOT::Math::IMultiGenFunction *getMultiGenFcn() = 0;

   RooMinimizer::Config const &cfg() const { return _context->_cfg; }

protected:
   void optimizeConstantTerms(bool constStatChange, bool constValChange);
   /// This function must be overridden in the derived class to pass on constant term optimization configuration
   /// to the function to be minimized. For a RooAbsArg, this would be RooAbsArg::constOptimizeTestStatistic.
   virtual void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) = 0;

   // used in BackProp (Minuit results -> RooFit) and ApplyCovarianceMatrix
   void SetPdfParamErr(Int_t index, double value);
   void ClearPdfParamAsymErr(Int_t index);
   void SetPdfParamErr(Int_t index, double loVal, double hiVal);

   void printEvalErrors() const;

   void finishDoEval() const;

   // members
   RooMinimizer *_context;

   // the following four are mutable because DoEval is const (in child classes)
   // Reset the *largest* negative log-likelihood value we have seen so far:
   mutable double _maxFCN = -std::numeric_limits<double>::infinity();
   mutable double _funcOffset{0.};
   mutable int _numBadNLL = 0;
   mutable int _evalCounter{0};

   unsigned int _nDim = 0;

   bool _optConst = false;

   std::unique_ptr<RooArgList> _floatParamList;
   std::unique_ptr<RooArgList> _constParamList;
   std::unique_ptr<RooArgList> _initFloatParamList;
   std::unique_ptr<RooArgList> _initConstParamList;

   std::ofstream *_logfile = nullptr;
};

#endif
