/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooFuncWrapper_h
#define RooFit_RooFuncWrapper_h

#include <RooAbsReal.h>
#include <RooListProxy.h>

#include <map>
#include <memory>
#include <string>
#include <sstream>

class RooSimultaneous;

namespace RooFit {

namespace Experimental {

/// @brief  A wrapper class to store a C++ function of type 'double (*)(double*, double*)'.
/// The parameters can be accessed as params[<relative position of param in paramSet>] in the function body.
/// The observables can be accessed as obs[i + j], where i represents the observable position and j
/// represents the data entry.
class RooFuncWrapper final : public RooAbsReal {
public:
   RooFuncWrapper(const char *name, const char *title, RooAbsReal &obj, const RooAbsData *data = nullptr,
                  RooSimultaneous const *simPdf = nullptr, bool useEvaluator = false);

   RooFuncWrapper(const RooFuncWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooFuncWrapper(*this, newname); }

   double defaultErrorLevel() const override { return 0.5; }

   bool hasGradient() const override { return _hasGradient; }
   void gradient(double *out) const override;

   void gradient(const double *x, double *g) const;

   std::size_t getNumParams() const { return _params.size(); }

   /// No constant term optimization is possible in code-generation mode.
   void constOptimizeTestStatistic(ConstOpCode /*opcode*/, bool /*doAlsoTrackingOpt*/) override {}

   std::string const &funcName() const { return _funcName; }

   void createGradient();

   void disableEvaluator() { _useEvaluator = false; }

   void writeDebugMacro(std::string const &) const;

   std::vector<std::string> const &collectedFunctions() { return _collectedFunctions; }

protected:
   double evaluate() const override;

private:
   void updateGradientVarBuffer() const;

   std::map<RooFit::Detail::DataKey, std::span<const double>>
   loadParamsAndData(RooArgSet const &paramSet, const RooAbsData *data, RooSimultaneous const *simPdf);

   void buildFuncAndGradFunctors();

   using Func = double (*)(double *, double const *, double const *);
   using Grad = void (*)(double *, double const *, double const *, double *);

   struct ObsInfo {
      ObsInfo(std::size_t i, std::size_t n) : idx{i}, size{n} {}
      std::size_t idx = 0;
      std::size_t size = 0;
   };

   std::unique_ptr<RooAbsReal> _absReal;
   RooListProxy _params;
   std::string _funcName;
   Func _func;
   Grad _grad;
   bool _hasGradient = false;
   bool _useEvaluator = false;
   mutable std::vector<double> _gradientVarBuffer;
   std::vector<double> _observables;
   std::map<RooFit::Detail::DataKey, ObsInfo> _obsInfos;
   std::vector<double> _xlArr;
   std::vector<std::string> _collectedFunctions;
};

} // namespace Experimental

} // namespace RooFit

#endif
