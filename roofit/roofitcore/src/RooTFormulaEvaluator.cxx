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
#include "RooFormulaUtils.h"
#include "RooMsgService.h"

#include "TFormula.h"

#include <sstream>

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
      // Construct a null-formula with all index placeholders replaced by a constant.
      TFormula nullFormula{"nullFormula",
                           RooFormulaUtils::reconstructFormula(processedFormula, varList, "1e-18").c_str(),
                           /*addToGlobList=*/false};
      const auto nullDim = nullFormula.GetNdim();
      if (nullDim != 0) {
         // TFormula thinks that we have an n-dimensional formula (n>0), but it shouldn't, as
         // these vars should have been replaced by constants in the null-formula
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
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor, copying the underlying TFormula.
RooTFormulaEvaluator::RooTFormulaEvaluator(RooTFormulaEvaluator const &other)
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

/// \endcond
