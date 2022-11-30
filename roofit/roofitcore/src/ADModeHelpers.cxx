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

#include "RooFit/ADModeHelpers.h"
#include "RooFit/BatchModeHelpers.h"

#include "RooFitDriver.h"
#include <TROOT.h>
#include <TSystem.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooBatchCompute.h>
#include <RooBinSamplingPdf.h>
#include <RooConstraintSum.h>
#include <RooDataSet.h>
#include "RooNLLVarNew.h"
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include "NormalizationHelpers.h"

#include <string>

using namespace ROOT::Experimental;

using namespace RooFit::BatchModeHelpers;

void RooFit::ADModeHelpers::BuildCodeRecur(RooAbsReal &head, std::string &code, std::string &global,
                                           std::unordered_map<const TNamed *, unsigned int> &paramVars,
                                           std::vector<std::string> &preFuncDecls)
{
   // First update the result variable of params in the compute graph to in[<position>].
   if (!head.isDerived()) {
      auto it = paramVars.find(head.namePtr());
      if (it != paramVars.end()) {
         head.updateResults("in[" + std::to_string(it->second) + "]");
      }
   }
   // then go ahead and build the final code.
   bool ILP = head.isReducerNode();
   if (ILP)
      code += head.buildLoopBegin(global);
   for (auto pcurr : head.servers()) {
      RooAbsReal *curr = dynamic_cast<RooAbsReal *>(pcurr);
      if (!curr) {
         std::stringstream errorMsg;
         errorMsg << "Translate is only supported for RooAbsReal derived objects.";
         oocoutE(nullptr, Minimization) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str().c_str());
      }
      BuildCodeRecur(*curr, code, global, paramVars, preFuncDecls);
   }
   code += head.translate(global, preFuncDecls);
   if (ILP)
      code += head.buildLoopEnd(global);
}

std::string RooFit::ADModeHelpers::BuildCode(RooAbsReal &arg,
                                             std::unordered_map<const TNamed *, unsigned int> &paramVars,
                                             std::string &globalConsts)
{
   std::string globalScope = "";
   std::string body = "";
   std::vector<std::string> funcDecls;
   RooFit::ADModeHelpers::BuildCodeRecur(arg, body, globalScope, paramVars, funcDecls);
   return "double func(double* in) {\n" + globalConsts + globalScope + body + "return " + arg.getResult() + ";\n}";
}

/// Function to unroll vectors into an array declarations into code.
void expandDecl(const std::string &name, std::vector<double> &values, std::string &outCode)
{
   std::string code = "double " + name + "[" + std::to_string(values.size()) + "]{";
   for (auto &it : values) {
      code += " " + std::to_string(it) + ",";
   }
   code.back() = '}';
   code += ";\n";
   outCode += code;
}

std::unique_ptr<RooAbsReal>
RooFit::ADModeHelpers::translateNLL(std::unique_ptr<RooAbsReal> &&obj, RooAbsData &data, bool printCode)
{
   std::unique_ptr<RooAbsRealWrapper> rooRealObj(static_cast<RooAbsRealWrapper *>(obj.release()));
   auto &topNode = rooRealObj->getRooFitDriverObj()->topNode();

   std::string globalConsts = "";
   unsigned int nEvents = data.numEntries();
   // Fixme: Hardcoded here right now.
   globalConsts += "unsigned int numEntries = " + std::to_string(nEvents) + ";\n";

   // Get the parameters from the observables to keep track of them in our 'in'
   // input array.
   std::unordered_map<const TNamed *, unsigned int> paramNames;
   RooArgSet observables{*data.get()};
   RooArgSet parameters;
   topNode.getParameters(&observables, parameters);
   for (RooAbsArg *arg : parameters) {
      paramNames[arg->namePtr()] = paramNames.size();
   }

   // Build declarations of observables into code.
   auto weight = data.getWeightBatch(0, nEvents, /*sumW2=*/false);
   if (weight.empty()) {
      std::vector<double> vals(nEvents, 1);
      expandDecl(RooNLLVarNew::weightVarName, vals, globalConsts);
   } else {
      std::vector<double> vals(nEvents);
      expandDecl(RooNLLVarNew::weightVarName, vals, globalConsts);
      for (std::size_t i = 0; i < nEvents; ++i)
         vals[i] = weight[i];
      expandDecl(RooNLLVarNew::weightVarName, vals, globalConsts);
   }

   for (auto const &item : data.getBatches(0, nEvents)) {
      RooSpan<const double> span{item.second};
      std::vector<double> vals(nEvents);
      for (std::size_t i = 0; i < nEvents; ++i) {
         vals[i] = span[i];
      }
      expandDecl(item.first->GetName(), vals, globalConsts);
   }

   // Actually build the code.
   std::string code = BuildCode(topNode, paramNames, globalConsts);

   std::string suffix = printCode ? "" : ";";
   bool comp = gCling->Declare(code.c_str());
   if (!comp) {
      std::stringstream errorMsg;
      errorMsg << "Translated code for AD could not be compiled. See above for details.";
      oocoutE(nullptr, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }
   auto funcObj = (double (*)(double *))gInterpreter->ProcessLine(("func" + suffix).c_str());

   // Return an instance of our wrapper class.
   return std::make_unique<RooGradFuncWrapper<decltype(funcObj)>>(funcObj, "func", parameters);
}
