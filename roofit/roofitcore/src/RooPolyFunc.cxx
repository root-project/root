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

using namespace std;
using namespace RooFit;

ClassImp(RooPolyFunc);

////////////////////////////////////////////////////////////////////////////////
/// coverity[UNINIT_CTOR]

void RooPolyFunc::addTerm(double coefficient)
{
   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = new RooRealVar(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   auto exponents = new RooArgList();

   for (const auto &var : _vars) {
      std::string exponent_name = Form("%s_%s^%d", GetName(), var->GetName(), 0);
      auto exponent = new RooRealVar(exponent_name.c_str(), exponent_name.c_str(), 0);
      exponents->add(*exponent);
   }

   termList->addOwned(*exponents);
   termList->addOwned(*coeff);
   this->_terms.push_back(move(termList));
}

void RooPolyFunc::addTerm(double coefficient, const RooAbsReal &var1, int exp1)
{
   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = new RooRealVar(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   auto exponents = new RooArgList();

   // linear iterate over all the variables, create var1^exp1 ..vark^0
   for (const auto &var : _vars) {
      int exp = 0;
      if (strcmp(var1.GetName(), var->GetName()) == 0)
         exp += exp1;
      std::string exponent_name = Form("%s_%s^%d", GetName(), var->GetName(), exp);
      auto exponent = new RooRealVar(exponent_name.c_str(), exponent_name.c_str(), exp);
      exponents->add(*exponent);
   }

   termList->addOwned(*exponents);
   termList->addOwned(*coeff);
   this->_terms.push_back(move(termList));
}

void RooPolyFunc::addTerm(double coefficient, const RooAbsReal &var1, int exp1, const RooAbsReal &var2, int exp2)
{

   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = new RooRealVar(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   auto exponents = new RooArgList();

   for (const auto &var : _vars) {
      int exp = 0;
      if (strcmp(var1.GetName(), var->GetName()) == 0)
         exp += exp1;
      if (strcmp(var2.GetName(), var->GetName()) == 0)
         exp += exp2;
      std::string exponent_name = Form("%s_%s^%d", GetName(), var->GetName(), exp);
      auto exponent = new RooRealVar(exponent_name.c_str(), exponent_name.c_str(), exp);
      exponents->add(*exponent);
   }
   termList->addOwned(*exponents);
   termList->addOwned(*coeff);
   this->_terms.push_back(move(termList));
}

void RooPolyFunc::addTerm(double coefficient, const RooAbsCollection &exponents)
{
   if (exponents.size() != _vars.size()) {
      coutE(InputArguments) << "RooPolyFunc::addTerm(" << GetName() << ") WARNING: number of exponents ("
                            << exponents.size() << ") provided do not match the number of variables (" << _vars.size()
                            << ")" << endl;
   }
   int n_terms = _terms.size();
   std::string coeff_name = Form("%s_c%d", GetName(), n_terms);
   std::string term_name = Form("%s_t%d", GetName(), n_terms);
   auto termList = std::make_unique<RooListProxy>(term_name.c_str(), term_name.c_str(), this);
   auto coeff = new RooRealVar(coeff_name.c_str(), coeff_name.c_str(), coefficient);
   termList->addOwned(exponents);
   termList->addOwned(*coeff);
   this->_terms.push_back(move(termList));
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooPolyFunc::RooPolyFunc() {}

////////////////////////////////////////////////////////////////////////////////
/// Parameterised constructor

RooPolyFunc::RooPolyFunc(const char *name, const char *title, const RooAbsCollection &vars)
   : RooAbsReal(name, title), _vars("x", "list of dependent variables", this)
{
   for (const auto &var : vars) {
      if (!dynamic_cast<RooAbsReal *>(var)) {
         std::stringstream ss;
         ss << "RooPolyFunc::ctor(" << GetName() << ") ERROR: coefficient " << var->GetName()
            << " is not of type RooAbsReal";
         const std::string errorMsg = ss.str();
         coutE(InputArguments) << errorMsg << std::endl;
         throw std::runtime_error(errorMsg);
      }
      _vars.add(*var);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooPolyFunc::RooPolyFunc(const RooPolyFunc &other, const char *name)
   : RooAbsReal(other, name), _vars("vars", this, other._vars)
{
   for (auto const &term : other._terms) {
      this->_terms.emplace_back(std::make_unique<RooListProxy>(term->GetName(), this, *term));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator

RooPolyFunc &RooPolyFunc::operator=(const RooPolyFunc &other)
{
   RooAbsReal::operator=(other);
   _vars = other._vars;

   for (auto const &term : other._terms) {
      this->_terms.emplace_back(std::make_unique<RooListProxy>(term->GetName(), this, *term));
   }
   return *this;
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
////////////////////////////////////////////////////////////////////////////////
///// Taylor expanding given function in terms of observables around
///// observableValues. Supports expansions upto order 2.
///// \param[in] function of variables that is taylor expanded.
///// \param[in] observables set of variables to perform the expansion.
///// \param[in] observableValues co-ordinates around which expansion is performed.
///// \param[in] order order of the expansion (0,1,2 supported).
///// \param[in] eps1 precision for first derivative and second derivative.
///// \param[in] eps2 precision for second partial derivative of cross-derivative.
std::unique_ptr<RooAbsReal>
RooPolyFunc::taylorExpand(const char *name, const char *title, RooAbsReal &func, const RooAbsCollection &observables,
                          std::vector<double> const &observableValues, int order, double eps1, double eps2)
{
   // create the taylor expansion polynomial
   auto taylor_poly = std::make_unique<RooPolyFunc>(name, title, observables);

   // taylor expansion can be performed only for order 0, 1, 2 currently
   if (order >= 3 || order <= 0) {
      std::stringstream errorMsgStream;
      errorMsgStream << "RooPolyFunc::taylorExpand(" << name << ") ERROR: order must be 0, 1, or 2";
      const auto errorMsg = errorMsgStream.str();
      oocoutE(taylor_poly.get(), InputArguments) << errorMsg << std::endl;
      throw std::invalid_argument(errorMsg);
   }

   setCoordinates(observables, observableValues);

   // estimate taylor expansion polynomial for different orders
   // f(x) = f(x=x0)
   //        + \sum_i + \frac{df}{dx_i}|_{x_i=x_i^0}(x - x_i^0)
   //        + \sum_{i,j} 0.5 * \frac{d^f}{dx_i x_j}
   //          ( x_i - x_i^0 )( x_j - x_j^0 )
   // note in the polynomial the () brackets are expanded out
   for (int i_order = 0; i_order <= order; ++i_order) {
      switch (i_order) {
      case 0: {
         taylor_poly->addTerm(func.getVal());
         break;
      }
      case 1: {
         for (auto *var : static_range_cast<RooRealVar *>(observables)) {
            double var1_val = var->getVal();
            auto deriv = func.derivative(*var, 1, eps2);
            double deriv_val = deriv->getVal();
            setCoordinates(observables, observableValues);
            taylor_poly->addTerm(deriv_val, *var, 1);
            if (var1_val != 0.0) {
               taylor_poly->addTerm(deriv_val * var1_val * -1.0);
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
                  setCoordinates(observables, observableValues);
               } else {
                  auto deriv2 = deriv1->derivative(*var2, 1, eps2);
                  deriv_val = 0.5 * deriv2->getVal();
                  setCoordinates(observables, observableValues);
               }
               taylor_poly->addTerm(deriv_val, *var1, 1, *var2, 1);
               if (var1_val != 0.0 || var2_val != 0.0) {
                  taylor_poly->addTerm(deriv_val * var1_val * var2_val);
                  taylor_poly->addTerm(deriv_val * var2_val * -1.0, *var1, 1);
                  taylor_poly->addTerm(deriv_val * var1_val * -1.0, *var2, 1);
               }
            }
         }
         break;
      }
      }
   }
   return taylor_poly;
}

////////////////////////////////////////////////////////////////////////////////
/// Taylor expanding given function in terms of observables around
/// defaultValue for all observables.
std::unique_ptr<RooAbsReal> RooPolyFunc::taylorExpand(const char *name, const char *title, RooAbsReal &func,
                                                      const RooAbsCollection &observables, double observablesValue,
                                                      int order, double eps1, double eps2)
{
   return RooPolyFunc::taylorExpand(name, title, func, observables,
                                    std::vector<double>(observables.size(), observablesValue), order, eps1, eps2);
}
