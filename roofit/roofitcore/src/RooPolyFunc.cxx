/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2021:                                                       *
 *      CERN, Switzerland                                                    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooPolyFunc
    \ingroup Roofit

RooPolyFunc implements a polynomial function in multi-variables.
The polynomial coefficients are implemented as doubles and are not
part of the RooFit computation graph.
**/

#include "RooPolyFunc.h"

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooArgSet.h"
#include "RooDerivative.h"
#include "RooMsgService.h"
#include "RooRealVar.h"

#include <utility>

using std::endl;
using namespace RooFit;


////////////////////////////////////////////////////////////////////////////////
/// coverity[UNINIT_CTOR]

void RooPolyFunc::addTerm(double coefficient)
{
   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = std::make_unique<RooRealVar>(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   RooArgList exponents{};

   for (const auto &var : _vars) {
      std::string exponent_name = Form("%s_%s^%d", GetName(), var->GetName(), 0);
      auto exponent = std::make_unique<RooRealVar>(exponent_name.c_str(), exponent_name.c_str(), 0);
      exponents.addOwned(std::move(exponent));
   }

   termList->addOwned(std::move(exponents));
   termList->addOwned(std::move(coeff));
   _terms.push_back(std::move(termList));
}

void RooPolyFunc::addTerm(double coefficient, const RooAbsReal &var1, int exp1)
{
   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = std::make_unique<RooRealVar>(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   RooArgList exponents{};

   // linear iterate over all the variables, create var1^exp1 ..vark^0
   for (const auto &var : _vars) {
      int exp = 0;
      if (strcmp(var1.GetName(), var->GetName()) == 0)
         exp += exp1;
      std::string exponent_name = Form("%s_%s^%d", GetName(), var->GetName(), exp);
      auto exponent = std::make_unique<RooRealVar>(exponent_name.c_str(), exponent_name.c_str(), exp);
      exponents.addOwned(std::move(exponent));
   }

   termList->addOwned(std::move(exponents));
   termList->addOwned(std::move(coeff));
   _terms.push_back(std::move(termList));
}

void RooPolyFunc::addTerm(double coefficient, const RooAbsReal &var1, int exp1, const RooAbsReal &var2, int exp2)
{

   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = std::make_unique<RooRealVar>(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   RooArgList exponents{};

   for (const auto &var : _vars) {
      int exp = 0;
      if (strcmp(var1.GetName(), var->GetName()) == 0)
         exp += exp1;
      if (strcmp(var2.GetName(), var->GetName()) == 0)
         exp += exp2;
      std::string exponent_name = Form("%s_%s^%d", GetName(), var->GetName(), exp);
      auto exponent = std::make_unique<RooRealVar>(exponent_name.c_str(), exponent_name.c_str(), exp);
      exponents.addOwned(std::move(exponent));
   }
   termList->addOwned(std::move(exponents));
   termList->addOwned(std::move(coeff));
   _terms.push_back(std::move(termList));
}

void RooPolyFunc::addTerm(double coefficient, const RooAbsCollection &exponents)
{
   if (exponents.size() != _vars.size()) {
      coutE(InputArguments) << "RooPolyFunc::addTerm(" << GetName() << ") WARNING: number of exponents ("
                            << exponents.size() << ") provided do not match the number of variables (" << _vars.size()
                            << ")" << std::endl;
   }
   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = std::make_unique<RooRealVar>(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   termList->addOwned(exponents);
   termList->addOwned(std::move(coeff));
   _terms.push_back(std::move(termList));
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooPolyFunc::RooPolyFunc() {}

////////////////////////////////////////////////////////////////////////////////
/// Parameterised constructor

RooPolyFunc::RooPolyFunc(const char *name, const char *title, const RooAbsCollection &vars)
   : RooAbsReal(name, title), _vars("x", "list of dependent variables", this)
{
   _vars.addTyped<RooAbsReal>(vars);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooPolyFunc::RooPolyFunc(const RooPolyFunc &other, const char *name)
   : RooAbsReal(other, name), _vars("vars", this, other._vars)
{
   for (auto const &term : other._terms) {
      _terms.emplace_back(std::make_unique<RooListProxy>(term->GetName(), this, *term));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return to RooPolyFunc as a string

std::string RooPolyFunc::asString() const
{
   std::stringstream ss;
   bool first = true;
   for (const auto &term : _terms) {
      size_t n_vars = term->size() - 1;
      auto coef = dynamic_cast<RooRealVar *>(term->at(n_vars));
      if (coef->getVal() > 0 && !first)
         ss << "+";
      ss << coef->getVal();
      first = true;
      for (size_t i_var = 0; i_var < n_vars; ++i_var) {
         auto var = dynamic_cast<RooRealVar *>(_vars.at(i_var));
         auto exp = dynamic_cast<RooRealVar *>(term->at(i_var));
         if (exp->getVal() == 0)
            continue;
         if (first) {
            ss << " * (";
         } else {
            ss << "*";
         }
         ss << "pow(" << var->GetName() << "," << exp->getVal() << ")";
         first = false;
      }
      if (!first)
         ss << ")";
   }
   return ss.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate value of Polynomial.
double RooPolyFunc::evaluate() const
{
   // Calculate and return value of polynomial
   double poly_sum(0.0);
   for (const auto &term : _terms) {
      double poly_term(1.0);
      size_t n_vars = term->size() - 1;
      for (size_t i_var = 0; i_var < n_vars; ++i_var) {
         auto var = dynamic_cast<RooRealVar *>(_vars.at(i_var));
         auto exp = dynamic_cast<RooRealVar *>(term->at(i_var));
         poly_term *= pow(var->getVal(), exp->getVal());
      }
      auto coef = dynamic_cast<RooRealVar *>(term->at(n_vars));
      poly_sum += coef->getVal() * poly_term;
   }
   return poly_sum;
}

void setCoordinates(const RooAbsCollection &observables, std::vector<double> const &observableValues)
// set all observables to expansion co-ordinate
{
   std::size_t iObs = 0;
   for (auto *var : static_range_cast<RooRealVar *>(observables)) {
      var->setVal(observableValues[iObs++]);
   }
}

void fixObservables(const RooAbsCollection &observables)
{
   for (auto *var : static_range_cast<RooRealVar *>(observables)) {
      var->setConstant(true);
   }
}

//////////////////////////////////////////////////////////////////////////////
/// Taylor expanding given function in terms of observables around
/// observableValues. Supports expansions upto order 2.
/// \param[in] name the name
/// \param[in] title the title
/// \param[in] func Function of variables that is taylor expanded.
/// \param[in] observables Set of variables to perform the expansion.
///            It's type is RooArgList to ensure that it is always ordered the
///            same as the observableValues vector. However, duplicate
///            observables are still not allowed.
/// \param[in] order Order of the expansion (0,1,2 supported).
/// \param[in] observableValues Coordinates around which expansion is
///            performed. If empty, the nominal observable values are taken, if
///            the size matches the size of the observables RooArgSet, the
///            values are mapped to the observables in matching order. If it
///            contains only one element, the same single value is used for all
///            observables.
/// \param[in] eps1 Precision for first derivative and second derivative.
/// \param[in] eps2 Precision for second partial derivative of cross-derivative.
std::unique_ptr<RooPolyFunc>
RooPolyFunc::taylorExpand(const char *name, const char *title, RooAbsReal &func, const RooArgList &observables,
                          int order, std::vector<double> const &observableValues, double eps1, double eps2)
{
   // Create the taylor expansion polynomial
   auto taylorPoly = std::make_unique<RooPolyFunc>(name, title, observables);

   // Verify that there are no duplicate observables
   {
      RooArgSet obsSet;
      for (RooAbsArg *obs : observables) {
         obsSet.add(*obs, /*silent*/ true); // we can be silent now, the error will come later
      }
      if (obsSet.size() != observables.size()) {
         std::stringstream errorMsgStream;
         errorMsgStream << "RooPolyFunc::taylorExpand(" << name << ") ERROR: duplicate input observables!";
         const auto errorMsg = errorMsgStream.str();
         oocoutE(taylorPoly.get(), InputArguments) << errorMsg << std::endl;
         throw std::invalid_argument(errorMsg);
      }
   }

   // Figure out the observable values around which to expand
   std::vector<double> obsValues;
   if (observableValues.empty()) {
      obsValues.reserve(observables.size());
      for (auto *var : static_range_cast<RooRealVar *>(observables)) {
         obsValues.push_back(var->getVal());
      }
   } else if (observableValues.size() == 1) {
      obsValues.resize(observables.size());
      std::fill(obsValues.begin(), obsValues.end(), observableValues[0]);
   } else if (observableValues.size() == observables.size()) {
      obsValues = observableValues;
   } else {
      std::stringstream errorMsgStream;
      errorMsgStream << "RooPolyFunc::taylorExpand(" << name
                     << ") ERROR: observableValues must be empty, contain one element, or match observables.size()!";
      const auto errorMsg = errorMsgStream.str();
      oocoutE(taylorPoly.get(), InputArguments) << errorMsg << std::endl;
      throw std::invalid_argument(errorMsg);
   }

   // taylor expansion can be performed only for order 0, 1, 2 currently
   if (order >= 3 || order <= 0) {
      std::stringstream errorMsgStream;
      errorMsgStream << "RooPolyFunc::taylorExpand(" << name << ") ERROR: order must be 0, 1, or 2";
      const auto errorMsg = errorMsgStream.str();
      oocoutE(taylorPoly.get(), InputArguments) << errorMsg << std::endl;
      throw std::invalid_argument(errorMsg);
   }

   setCoordinates(observables, obsValues);

   // estimate taylor expansion polynomial for different orders
   // f(x) = f(x=x0)
   //        + \sum_i + \frac{df}{dx_i}|_{x_i=x_i^0}(x - x_i^0)
   //        + \sum_{i,j} 0.5 * \frac{d^f}{dx_i x_j}
   //          ( x_i - x_i^0 )( x_j - x_j^0 )
   // note in the polynomial the () brackets are expanded out
   for (int i_order = 0; i_order <= order; ++i_order) {
      switch (i_order) {
      case 0: {
         taylorPoly->addTerm(func.getVal());
         break;
      }
      case 1: {
         for (auto *var : static_range_cast<RooRealVar *>(observables)) {
            double var1_val = var->getVal();
            auto deriv = func.derivative(*var, 1, eps2);
            double deriv_val = deriv->getVal();
            setCoordinates(observables, obsValues);
            taylorPoly->addTerm(deriv_val, *var, 1);
            if (var1_val != 0.0) {
               taylorPoly->addTerm(deriv_val * var1_val * -1.0);
            }
         }
         break;
      }
      case 2: {
         for (auto *var1 : static_range_cast<RooRealVar *>(observables)) {
            double var1_val = var1->getVal();
            auto deriv1 = func.derivative(*var1, 1, eps1);
            for (auto *var2 : static_range_cast<RooRealVar *>(observables)) {
               double var2_val = var2->getVal();
               double deriv_val = 0.0;
               if (strcmp(var1->GetName(), var2->GetName()) == 0) {
                  auto deriv2 = func.derivative(*var2, 2, eps2);
                  deriv_val = 0.5 * deriv2->getVal();
                  setCoordinates(observables, obsValues);
               } else {
                  auto deriv2 = deriv1->derivative(*var2, 1, eps2);
                  deriv_val = 0.5 * deriv2->getVal();
                  setCoordinates(observables, obsValues);
               }
               taylorPoly->addTerm(deriv_val, *var1, 1, *var2, 1);
               if (var1_val != 0.0 || var2_val != 0.0) {
                  taylorPoly->addTerm(deriv_val * var1_val * var2_val);
                  taylorPoly->addTerm(deriv_val * var2_val * -1.0, *var1, 1);
                  taylorPoly->addTerm(deriv_val * var1_val * -1.0, *var2, 1);
               }
            }
         }
         break;
      }
      }
   }
   return taylorPoly;
}
