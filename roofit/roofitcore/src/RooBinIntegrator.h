/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_BIN_INTEGRATOR
#define ROO_BIN_INTEGRATOR

#include "RooAbsIntegrator.h"
#include "RooNumIntConfig.h"
#include "Math/Util.h"
#include <vector>

class RooBinIntegrator : public RooAbsIntegrator {
public:
   // Constructors, assignment etc

   RooBinIntegrator(const RooAbsFunc &function, int numBins = 100);
   RooBinIntegrator(const RooAbsFunc &function, const RooNumIntConfig &config);

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
   RooBinIntegrator(const RooBinIntegrator &);

   // Numerical integrator workspace
   mutable std::vector<double> _xmin;      ///<! Lower integration bound
   mutable std::vector<double> _xmax;      ///<! Upper integration bound
   std::vector<std::vector<double>> _binb; ///<! list of bin boundaries
   mutable Int_t _numBins = 0;             ///<! Size of integration range

   bool _useIntegrandLimits = false; ///< If true limits of function binding are used

   void recursive_integration(const UInt_t d, const double delta, ROOT::Math::KahanSum<double>& sum);

   std::vector<double> _x; ///<! do not persist, store here N-dimensional evaluation point when integrating
};

#endif
