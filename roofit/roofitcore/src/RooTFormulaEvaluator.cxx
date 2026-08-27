/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooTFormulaEvaluator.h"

#include "RooArgList.h"
#include "RooMsgService.h"

#include "TFormula.h"

#include <cassert>
#include <regex>
#include <sstream>

namespace {

////////////////////////////////////////////////////////////////////////////////
/// From the internal representation, construct a null-formula by replacing all
/// index place holders with zeroes, and return it as string
std::string reconstructNullFormula(std::string internalRepr, RooArgList const &args)
{
   const auto nArgs = args.size();
   for (unsigned int i = 0; i < nArgs; ++i) {
      std::stringstream regexStr;
      regexStr << "x\\[" << i << "\\]|@" << i;
      std::regex regex(regexStr.str());

      std::string replacement = "1e-18";
      internalRepr = std::regex_replace(internalRepr, regex, replacement);
   }

   return internalRepr;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Construct the TFormula from the processed formula string, checking that it
/// compiles and also fulfills the assumptions. Throws otherwise, with the
/// original formula string appearing in the error messages.
RooTFormulaEvaluator::RooTFormulaEvaluator(const char *name, std::string const &processedFormula,
                                           std::string const &origFormula, RooArgList const &varList)
{
   auto theFormula = std::make_unique<TFormula>(name, processedFormula.c_str(), /*addToGlobList=*/false);

   if (!theFormula->IsValid()) {
      std::stringstream msg;
      msg << "RooFormula '" << name << "' did not compile or is invalid."
          << "\nInput:\n\t" << origFormula << "\nPassed over to TFormula:\n\t" << processedFormula << std::endl;
      oocoutF(static_cast<TObject *>(nullptr), InputArguments) << msg.str();
      throw std::runtime_error(msg.str());
   }

   if (theFormula->GetNdim() != 0) {
      TFormula nullFormula{"nullFormula", reconstructNullFormula(processedFormula, varList).c_str(),
                           /*addToGlobList=*/false};
      const auto nullDim = nullFormula.GetNdim();
      if (nullDim != 0) {
         // TFormula thinks that we have an n-dimensional formula (n>0), but it shouldn't, as
         // these vars should have been replaced by zeroes in reconstructNullFormula
         // since RooFit only uses the syntax x[0], x[1], x[2], ...
         // This can happen e.g. with variables x,y,z,t that were not supplied in arglist.
         std::stringstream msg;
         msg << "TFormula interprets the formula " << origFormula << " as " << theFormula->GetNdim() + nullDim
             << "-dimensional with undefined variable(s) {";
         for (auto i = 0; i < nullDim; ++i) {
            msg << nullFormula.GetVarName(i) << ",";
         }
         msg << "}, which could not be supplied by RooFit."
             << "\nThe formula must be modified, or those variables must be supplied in the list of variables."
             << std::endl;
         oocoutF(static_cast<TObject *>(nullptr), InputArguments) << msg.str();
         throw std::invalid_argument(msg.str());
      }
   }

   _tFormula = std::move(theFormula);

   // Find out which variables `x[i]` are actually referenced in the formula.
   _varIsUsed.resize(varList.size());
   static const std::regex newOrdinalRegex("\\bx\\[([0-9]+)\\]");
   for (auto matchIt = std::sregex_iterator(processedFormula.begin(), processedFormula.end(), newOrdinalRegex);
        matchIt != std::sregex_iterator(); ++matchIt) {
      assert(matchIt->size() == 2);
      std::stringstream matchString((*matchIt)[1]);
      unsigned int i;
      matchString >> i;

      if (i < _varIsUsed.size()) {
         _varIsUsed[i] = true;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor, copying the underlying TFormula.
RooTFormulaEvaluator::RooTFormulaEvaluator(RooTFormulaEvaluator const &other) : _varIsUsed{other._varIsUsed}
{
   if (other._tFormula) {
      _tFormula = std::make_unique<TFormula>(*other._tFormula);
   }
}

RooTFormulaEvaluator::~RooTFormulaEvaluator() = default;

////////////////////////////////////////////////////////////////////////////////
/// Evaluate the internal TFormula.
double RooTFormulaEvaluator::eval(const double *vars) const
{
   return _tFormula->EvalPar(vars);
}

std::unique_ptr<RooFormulaEvaluator> RooTFormulaEvaluator::clone() const
{
   return std::make_unique<RooTFormulaEvaluator>(*this);
}

bool RooTFormulaEvaluator::usesVariable(unsigned int i) const
{
   return i < _varIsUsed.size() && _varIsUsed[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Return the processed formula string, which is the title of the TFormula.
std::string RooTFormulaEvaluator::processedFormula() const
{
   return _tFormula->GetTitle();
}

/// \endcond
