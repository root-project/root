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

#include <memory>
#include <string>

/// @brief  A wrapper class to store a C++ function of type 'double (*)(double*, double*)'.
/// The parameters can be accessed as params[<relative position of param in paramSet>] in the function body.
/// The observables can be accessed as obs[i + j], where i represents the observable position and j
/// represents the data entry.
class RooFuncWrapper final : public RooAbsReal {
public:
   RooFuncWrapper(const char *name, const char *title, std::string const &funcBody, RooArgSet const &paramSet,
                  RooArgSet const &obsSet, const RooAbsData *data = nullptr);

   RooFuncWrapper(const char *name, const char *title, RooAbsReal const &obj, RooArgSet const &normSet,
                  const RooAbsData *data = nullptr);

   RooFuncWrapper(const RooFuncWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooFuncWrapper(*this, newname); }

   double defaultErrorLevel() const override { return 0.5; }

   void getGradient(double *out) const;

   void gradient(const double *x, double *g) const;

   std::size_t getNumParams() const { return _params.size(); }

protected:
   double evaluate() const override;

private:
   std::string
   buildCode(RooAbsReal const &head, RooArgSet const & /* paramSet */, RooArgSet const &obsSet, const RooAbsData *data);

   void updateGradientVarBuffer() const;

   void
   loadParamsAndObs(std::string funcName, RooArgSet const &paramSet, RooArgSet const &obsSet, const RooAbsData *data);

   void declareAndDiffFunction(std::string funcName, std::string const &funcBody);

   void buildFuncAndGradFunctors();

   using Func = double (*)(double *, double *);
   using Grad = void (*)(double *, double *, double *);

   RooListProxy _params;
   Func _func;
   Grad _grad;
   mutable std::vector<double> _gradientVarBuffer;
   mutable std::vector<double> _observables;
};

#endif
