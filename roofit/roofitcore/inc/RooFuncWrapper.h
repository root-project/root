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

#include "TROOT.h"
#include "TSystem.h"
#include "RooAbsReal.h"
#include "RooGlobalFunc.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooListProxy.h"

#include <memory>
#include <string>

/// @brief  A wrapper class to store a C++ function of type 'double (*)(double* )'.
/// The parameters can be accessed as x[<relative position of param in paramSet>] in the function body.
/// @tparam Func Function pointer to the generated function.
template <typename Func = double (*)(double *), typename Grad = void (*)(double *, double *)>
class RooFuncWrapper final : public RooAbsReal {
public:
   RooFuncWrapper(const char *name, const char *title, std::string const &funcBody, RooArgSet const &paramSet)
      : RooAbsReal{name, title}, _params{"!params", "List of parameters", this}
   {
      std::string funcName = name;
      std::string gradName = funcName + "_grad";
      std::string requestName = funcName + "_req";
      std::string wrapperName = funcName + "_derivativeWrapper";

      gInterpreter->Declare("#pragma cling optimize(2)");

      // Declare the function
      std::stringstream bodyWithSigStrm;
      bodyWithSigStrm << "double " << funcName << "(double* x) {" << funcBody << "}";
      bool comp = gInterpreter->Declare(bodyWithSigStrm.str().c_str());
      if (!comp) {
         std::stringstream errorMsg;
         errorMsg << "Function " << funcName << " could not be compiled. See above for details.";
         coutE(InputArguments) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str().c_str());
      }
      _func = reinterpret_cast<Func>(gInterpreter->ProcessLine((funcName + ";").c_str()));

      // Calculate gradient
      gInterpreter->ProcessLine("#include <Math/CladDerivator.h>");
      // disable clang-format for making the following code unreadable.
      // clang-format off
      std::stringstream requestFuncStrm;
      requestFuncStrm << "#pragma clad ON\n"
                         "void " << requestName << "() {\n"
                         "  clad::gradient(" << funcName << ");\n"
                         "}\n"
                         "#pragma clad OFF";
      // clang-format on
      comp = gInterpreter->Declare(requestFuncStrm.str().c_str());
      if (!comp) {
         std::stringstream errorMsg;
         errorMsg << "Function " << funcName << " could not be be differentiated. See above for details.";
         coutE(InputArguments) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str().c_str());
      }

      // Extract parameters
      for (auto *param : paramSet) {
         if (!dynamic_cast<RooAbsReal *>(param)) {
            std::stringstream errorMsg;
            errorMsg << "In creation of function " << funcName
                     << " wrapper: input param expected to be of type RooAbsReal.";
            coutE(InputArguments) << errorMsg.str() << std::endl;
            throw std::runtime_error(errorMsg.str().c_str());
         }
         _params.add(*param);
      }
      _gradientVarBuffer.reserve(_params.size());

      // Build a wrapper over the derivative to hide clad specific types such as 'array_ref'.
      // disable clang-format for making the following code unreadable.
      // clang-format off
      std::stringstream dWrapperStrm;
      dWrapperStrm << "void " << wrapperName << "(double* in, double* out) {\n"
                      "  clad::array_ref<double> cladOut(out, " << _params.size() << ");\n"
                      "  " << gradName << "(in, cladOut);\n"
                      "}";
      // clang-format on
      gInterpreter->Declare(dWrapperStrm.str().c_str());
      _grad = reinterpret_cast<Grad>(gInterpreter->ProcessLine((wrapperName + ";").c_str()));
   }

   RooFuncWrapper(const RooFuncWrapper &other, const char *name = nullptr)
      : RooAbsReal(other, name), _params("!params", this, other._params), _func(other._func)
   {
   }

   TObject *clone(const char *newname) const override { return new RooFuncWrapper(*this, newname); }

   double defaultErrorLevel() const override { return 0.5; }

   void getGradient(double *out) const
   {
      updateGradientVarBuffer();

      _grad(_gradientVarBuffer.data(), out);
   }

protected:
   void updateGradientVarBuffer() const
   {
      std::transform(_params.begin(), _params.end(), _gradientVarBuffer.begin(),
                     [](RooAbsArg *obj) { return static_cast<RooAbsReal *>(obj)->getValV(); });
   }

   double evaluate() const override
   {
      updateGradientVarBuffer();

      return _func(_gradientVarBuffer.data());
   }

private:
   RooListProxy _params;
   Func _func;
   Grad _grad;
   mutable std::vector<double> _gradientVarBuffer;
};

#endif
