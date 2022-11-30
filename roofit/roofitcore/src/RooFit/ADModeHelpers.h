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

#ifndef RooFit_ADModeHelpers_h
#define RooFit_ADModeHelpers_h

#include <RooGlobalFunc.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooListProxy.h>

#include <memory>
#include <string>

class RooAbsData;
class RooAbsPdf;
class RooArgSet;

/// @brief  A wrapper class to store the generated function of a given RooFit workspace.
/// @tparam Func Function pointer to the generated function.
template <typename Func = void>
class RooGradFuncWrapper final : public RooAbsReal {
public:
   RooGradFuncWrapper(Func function, std::string funcName, RooArgSet const &paramSet)
      : RooAbsReal{"RooGradFuncWrapper", "RooGradFuncWrapper"}, _paramProxies{"!params", "List of parameters", this},
        _funcName(funcName), _func(function)
   {
      for (auto *param : paramSet) {
         _paramProxies.add(*param);
         if (auto realParam = dynamic_cast<RooRealVar *>(param))
            _params.emplace_back(realParam);
      }
   }

   RooGradFuncWrapper(const RooGradFuncWrapper &other, const char *name = nullptr)
      : RooAbsReal(other, name), _paramProxies("!params", this, other._paramProxies), _func(other._func)
   {
   }

   TObject *clone(const char *newname) const override { return new RooGradFuncWrapper(*this, newname); }

   double defaultErrorLevel() const override { return 0.5; }

protected:
   double evaluate() const override
   {
      std::vector<double> paramVals;
      paramVals.reserve(_paramProxies.size());
      std::transform(_params.begin(), _params.end(), std::back_inserter(paramVals),
                     [](RooAbsReal *obj) { return obj->getValV(); });

      return _func(paramVals.data());
   }

private:
   RooListProxy _paramProxies;
   std::vector<RooRealVar *> _params;
   std::string _funcName;
   Func _func;
};

namespace RooFit {
namespace ADModeHelpers {

std::string
BuildCode(RooAbsReal &arg, std::unordered_map<const TNamed *, unsigned int> &paramVars, std::string &globalConsts);

void BuildCodeRecur(RooAbsReal &arg, std::string &code, std::string &global,
                    std::unordered_map<const TNamed *, unsigned int> &paramVars,
                    std::vector<std::string> &preFuncDecls);

std::unique_ptr<RooAbsReal> translateNLL(std::unique_ptr<RooAbsReal> &&obj, RooAbsData &data, bool printCode);

} // namespace ADModeHelpers
} // namespace RooFit

#endif
