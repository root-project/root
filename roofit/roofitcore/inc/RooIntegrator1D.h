/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooIntegrator1D.h,v 1.21 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_INTEGRATOR_1D
#define ROO_INTEGRATOR_1D

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"

#include <ROOT/RSpan.hxx>

#include <functional>

namespace RooFit {
namespace Detail {

std::pair<double, int> integrate1d(std::function<double(double)> func, bool doTrapezoid, int maxSteps, int minStepsZero,
                                   int fixSteps, double epsAbs, double epsRel, bool doExtrap, double xmin, double xmax,
                                   std::span<double> hArr, std::span<double> sArr);

} // namespace Detail
} // namespace RooFit

class RooIntegrator1D : public RooAbsIntegrator {
public:
   // Constructors, assignment etc
   enum SummationRule { Trapezoid, Midpoint };

   RooIntegrator1D(const RooAbsFunc &function, SummationRule rule = Trapezoid, int maxSteps = 0, double eps = 0);
   RooIntegrator1D(const RooAbsFunc &function, double xmin, double xmax, SummationRule rule = Trapezoid,
                   int maxSteps = 0, double eps = 0);

   RooIntegrator1D(const RooAbsFunc &function, const RooNumIntConfig &config, int nDim = 1);
   RooIntegrator1D(const RooAbsFunc &function, double xmin, double xmax, const RooNumIntConfig &config, int nDim = 1);

   bool checkLimits() const override;
   double integral(const double *yvec = nullptr) override;

   using RooAbsIntegrator::setLimits;
   bool setLimits(double *xmin, double *xmax) override;
   bool setUseIntegrandLimits(bool flag) override
   {
      _useIntegrandLimits = flag;
      return true;
   }

protected:
   friend class RooNumIntFactory;
   static void registerIntegrator(RooNumIntFactory &fact);

   bool initialize();
   double integral(const double *yvec, int iDim, std::span<double> wksp);

   bool _useIntegrandLimits; ///< If true limits of function binding are used

   // Integrator configuration
   int _nDim = 1;
   SummationRule _rule;
   int _maxSteps;             ///< Maximum number of steps
   int _minStepsZero = 999;   ///< Minimum number of steps to declare convergence to zero
   int _fixSteps = 0;         ///< Fixed number of steps
   double _epsAbs;            ///< Absolute convergence tolerance
   double _epsRel;            ///< Relative convergence tolerance
   bool _doExtrap = true;     ///< Apply conversion step?
   std::vector<double> _xmin; ///<! Lower integration bounds
   std::vector<double> _xmax; ///<! Upper integration bounds

   // Numerical integrator workspace
   std::vector<double> _wksp; ///<! Integrator workspace
   std::vector<double> _x;    //! do not persist

   ClassDefOverride(RooIntegrator1D, 0) // 1-dimensional numerical integration engine
};

#endif
