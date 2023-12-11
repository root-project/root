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

class RooSimultaneous;

/// @brief  A wrapper class to store a C++ function of type 'double (*)(double*, double*)'.
/// The parameters can be accessed as params[<relative position of param in paramSet>] in the function body.
/// The observables can be accessed as obs[i + j], where i represents the observable position and j
/// represents the data entry.
class RooFuncWrapper final : public RooAbsReal {
public:
   RooFuncWrapper(const char *name, const char *title, RooAbsReal const &obj, RooArgSet const &normSet,
                  const RooAbsData *data, RooSimultaneous const *simPdf, bool createGradient);

   RooFuncWrapper(const RooFuncWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooFuncWrapper(*this, newname); }

   double defaultErrorLevel() const override { return 0.5; }

   bool hasGradient() const override { return _hasGradient; }
   void gradient(double *out) const override;

   void gradient(const double *x, double *g) const;

   std::size_t getNumParams() const { return _params.size(); }

   void dumpCode();

   void dumpGradient();

   /// No constant term optimization is possible in code-generation mode.
   void constOptimizeTestStatistic(ConstOpCode /*opcode*/, bool /*doAlsoTrackingOpt*/) override {}

   std::string const &funcName() const { return _funcName; }

protected:
   double evaluate() const override;

private:
   std::string buildCode(RooAbsReal const &head);

   void updateGradientVarBuffer() const;

   void loadParamsAndData(RooAbsArg const *head, RooArgSet const &paramSet, const RooAbsData *data,
                          RooSimultaneous const *simPdf);

   void declareAndDiffFunction(std::string const &funcBody, bool createGradient);

   void buildFuncAndGradFunctors();

   using Func = double (*)(double *, double const *);
   using Grad = void (*)(double *, double const *, double *);

   struct ObsInfo {
      ObsInfo(std::size_t i, std::size_t n) : idx{i}, size{n} {}
      std::size_t idx = 0;
      std::size_t size = 0;
   };

   RooListProxy _params;
   std::string _funcName;
   Func _func;
   Grad _grad;
   bool _hasGradient = false;
   mutable std::vector<double> _gradientVarBuffer;
   std::vector<double> _observables;
   std::map<RooFit::Detail::DataKey, ObsInfo> _obsInfos;
   std::map<RooFit::Detail::DataKey, std::size_t> _nodeOutputSizes;
};

#endif
