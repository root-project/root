/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_INTEGRATOR_BINDING
#define ROO_INTEGRATOR_BINDING

#include <RooAbsFunc.h>
#include <RooAbsIntegrator.h>

// Function binding representing output of numeric integrator.
class RooIntegratorBinding : public RooAbsFunc {
public:
   RooIntegratorBinding(std::unique_ptr<RooAbsIntegrator> integrator)
      : RooAbsFunc(integrator->integrand()->getDimension() - 1), _integrator(integrator.get())
   {
   }

   RooAbsIntegrator const &integrator() const { return *_integrator; }

   inline double operator()(const double xvector[]) const override
   {
      _ncall++;
      return _integrator->integral(xvector);
   }
   inline double getMinLimit(UInt_t index) const override { return _integrator->integrand()->getMinLimit(index + 1); }
   inline double getMaxLimit(UInt_t index) const override { return _integrator->integrand()->getMaxLimit(index + 1); }

private:
   std::unique_ptr<RooAbsIntegrator> _integrator; ///< Numeric integrator
};

#endif
