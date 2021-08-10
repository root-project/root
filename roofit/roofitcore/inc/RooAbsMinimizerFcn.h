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
#include "RooRealVar.h"

#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <memory>  // unique_ptr

// forward declaration
class RooMinimizer;

class RooAbsMinimizerFcn {

public:

   // Adapter class to define the necessary interface and data required by the RooAbsMinimizerFcn.
   struct Function {
       std::function<double()> getVal;
       std::function<void(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt)> constOptimizeTestStatistic;
       const std::string name;
       const std::string title;
       RooArgSet parameters;
       double errorLevel;

       Function(RooAbsReal & absReal)
         : getVal([&absReal](){ return absReal.getVal(); })
         , constOptimizeTestStatistic([&absReal](RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt){
                 absReal.constOptimizeTestStatistic(opcode, doAlsoTrackingOpt); })
         , name{absReal.GetName()}
         , title{absReal.GetTitle()}
         , errorLevel{absReal.defaultErrorLevel()}
       {
          absReal.getParameters(nullptr, parameters);
       }
   };

   RooAbsMinimizerFcn(RooArgList paramList, RooMinimizer *context, bool verbose = false);
   RooAbsMinimizerFcn(const RooAbsMinimizerFcn &other);
   virtual ~RooAbsMinimizerFcn() = default;

   /// Informs Minuit through its parameter_settings vector of RooFit parameter properties.
   Bool_t synchronizeParameterSettings(std::vector<ROOT::Fit::ParameterSettings> &parameters, Bool_t optConst, Bool_t verbose);
   /// Like synchronizeParameterSettings, Synchronize informs Minuit through its parameter_settings vector of RooFit parameter properties,
   /// but Synchronize can be overridden to e.g. also include gradient strategy synchronization in subclasses.
   virtual Bool_t Synchronize(std::vector<ROOT::Fit::ParameterSettings> &parameters, Bool_t optConst, Bool_t verbose);

   RooArgList *GetFloatParamList();
   RooArgList *GetConstParamList();
   RooArgList *GetInitFloatParamList();
   RooArgList *GetInitConstParamList();
   Int_t GetNumInvalidNLL() const;

   // need access from Minimizer:
   void SetEvalErrorWall(Bool_t flag);
   /// Try to recover from invalid function values. When invalid function values are encountered,
   /// a penalty term is returned to the minimiser to make it back off. This sets the strength of this penalty.
   /// \note A strength of zero is equivalent to a constant penalty (= the gradient vanishes, ROOT < 6.24).
   /// Positive values lead to a gradient pointing away from the undefined regions. Use ~10 to force the minimiser
   /// away from invalid function values.
   void SetRecoverFromNaNStrength(double strength) { _recoverFromNaNStrength = strength; }
   void SetPrintEvalErrors(Int_t numEvalErrors);
   Double_t &GetMaxFCN();
   Int_t evalCounter() const;
   void zeroEvalCount();
   /// Return a possible offset that's applied to the function to separate invalid function values from valid ones.
   double getOffset() const { return _funcOffset; }
   void SetVerbose(Bool_t flag = kTRUE);

   /// Put Minuit results back into RooFit objects.
   void BackProp(const ROOT::Fit::FitResult &results);

   /// RooMinimizer sometimes needs the name of the minimized function. Implement this in the derived class.
   virtual std::string const& getFunctionName() const = 0;
   /// RooMinimizer sometimes needs the title of the minimized function. Implement this in the derived class.
   virtual std::string const& getFunctionTitle() const = 0;

   /// Set different external covariance matrix
   void ApplyCovarianceMatrix(TMatrixDSym &V);

   Bool_t SetLogFile(const char *inLogfile);
   std::ofstream *GetLogFile() { return _logfile; }

   unsigned int getNDim() const { return _nDim; }

   void setOptimizeConst(Int_t flag);

   bool getOptConst();
   std::vector<double> getParameterValues() const;

   Bool_t SetPdfParamVal(int index, double value) const;

protected:
   void optimizeConstantTerms(bool constStatChange, bool constValChange);
   virtual Function & funct() = 0;

   // used in BackProp (Minuit results -> RooFit) and ApplyCovarianceMatrix
   void SetPdfParamErr(Int_t index, Double_t value);
   void ClearPdfParamAsymErr(Int_t index);
   void SetPdfParamErr(Int_t index, Double_t loVal, Double_t hiVal);

   void printEvalErrors() const;

   // members
   RooMinimizer *_context;

   // the following four are mutable because DoEval is const (in child classes)
   // Reset the *largest* negative log-likelihood value we have seen so far:
   mutable double _maxFCN = -std::numeric_limits<double>::infinity();
   mutable double _funcOffset{0.};
   double _recoverFromNaNStrength{10.};
   mutable int _numBadNLL = 0;
   mutable int _printEvalErrors = 10;
   mutable int _evalCounter{0};

   unsigned int _nDim = 0;

   Bool_t _optConst = kFALSE;

   std::unique_ptr<RooArgList> _floatParamList;
   std::unique_ptr<RooArgList> _constParamList;
   std::unique_ptr<RooArgList> _initFloatParamList;
   std::unique_ptr<RooArgList> _initConstParamList;

   std::ofstream *_logfile = nullptr;
   bool _doEvalErrorWall{true};
   bool _verbose;
};

#endif
