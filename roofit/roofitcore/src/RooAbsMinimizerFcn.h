/// \cond ROOFIT_INTERNAL

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

#include "TMatrixDSym.h"

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooMinimizer.h"
#include "RooRealVar.h"

#include <iostream>
#include <fstream>
#include <string>
#include <memory> // unique_ptr

class RooAbsMinimizerFcn {

public:
   RooAbsMinimizerFcn(RooArgList paramList, RooMinimizer *context);
   virtual ~RooAbsMinimizerFcn() = default;

   /// Informs Minuit through its parameter_settings vector of RooFit parameter properties.
   bool synchronizeParameterSettings(std::vector<ROOT::Fit::ParameterSettings> &parameters, bool optConst);
   /// Like synchronizeParameterSettings, Synchronize informs Minuit through
   /// its parameter_settings vector of RooFit parameter properties, but
   /// Synchronize can be overridden to e.g. also include gradient strategy
   /// synchronization in subclasses.
   virtual bool Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters);

   RooArgList const &allParams() const { return _allParams; }
   RooArgList floatParams() const;
   RooArgList constParams() const;
   RooArgList initFloatParams() const;
   Int_t GetNumInvalidNLL() const { return _numBadNLL; }

   double &GetMaxFCN() { return _maxFCN; }
   Int_t evalCounter() const { return _evalCounter; }
   void zeroEvalCount() { _evalCounter = 0; }
   /// Return a possible offset that's applied to the function to separate invalid function values from valid ones.
   double &getOffset() const { return _funcOffset; }

   /// Put Minuit results back into RooFit objects.
   void BackProp();

   /// RooMinimizer sometimes needs the name of the minimized function. Implement this in the derived class.
   virtual std::string getFunctionName() const = 0;
   /// RooMinimizer sometimes needs the title of the minimized function. Implement this in the derived class.
   virtual std::string getFunctionTitle() const = 0;

   /// Set different external covariance matrix
   void ApplyCovarianceMatrix(TMatrixDSym &V);

   bool SetLogFile(const char *inLogfile);
   std::ofstream *GetLogFile() { return _logfile; }

   unsigned int getNDim() const { return _floatableParamIndices.size(); }

   void setOptimizeConst(Int_t flag);

   bool SetPdfParamVal(int index, double value) const;

   /// Enable or disable offsetting on the function to be minimized, which enhances numerical precision.
   virtual void setOffsetting(bool flag) = 0;
   virtual ROOT::Math::IMultiGenFunction *getMultiGenFcn() = 0;

   RooMinimizer::Config const &cfg() const { return _context->_cfg; }

   inline RooRealVar &floatableParam(std::size_t i) const
   {
      return static_cast<RooRealVar &>(_allParams[_floatableParamIndices[i]]);
   }

protected:
   void optimizeConstantTerms(bool constStatChange, bool constValChange);
   /// This function must be overridden in the derived class to pass on constant term optimization configuration
   /// to the function to be minimized. For a RooAbsArg, this would be RooAbsArg::constOptimizeTestStatistic.
   virtual void setOptimizeConstOnFunction(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) = 0;

   void printEvalErrors() const;

   double applyEvalErrorHandling(double fvalue) const;
   void finishDoEval() const;

   inline static bool canBeFloating(RooAbsArg const &arg) { return dynamic_cast<RooRealVar const *>(&arg); }

   // Figure out whether we have to treat this parameter as a constant.
   inline static bool treatAsConstant(RooAbsArg const &arg) { return arg.isConstant() || !canBeFloating(arg); }

   // members
   RooMinimizer *_context = nullptr;

   // the following four are mutable because DoEval is const (in child classes)
   // Reset the *largest* negative log-likelihood value we have seen so far:
   mutable double _maxFCN = -std::numeric_limits<double>::infinity();
   mutable double _funcOffset{0.};
   mutable int _numBadNLL = 0;
   mutable int _evalCounter{0};
   // PB: these mutables signal a suboptimal design. A separate error handling
   // object containing all this would clean up this class. It would allow const
   // functions to be actually const (even though state still changes in the
   // error handling object).

   bool _optConst = false;

   RooArgList _allParams;
   RooArgList _allParamsInit;

   std::vector<std::size_t> _floatableParamIndices;

   std::ofstream *_logfile = nullptr;
};

#endif

/// \endcond
