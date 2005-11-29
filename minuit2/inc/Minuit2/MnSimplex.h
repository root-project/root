// @(#)root/minuit2:$Name:  $:$Id: MnSimplex.h,v 1.2.6.3 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnSimplex
#define ROOT_Minuit2_MnSimplex

#include "Minuit2/MnApplication.h"
#include "Minuit2/SimplexMinimizer.h"

namespace ROOT {

   namespace Minuit2 {


class FCNBase;

/** API class for minimization using Variable Metric technology ("MIGRAD");
    allows for user interaction: set/change parameters, do minimization,
    change parameters, re-do minimization etc.; 
    also used by MnMinos and MnContours;
 */

class MnSimplex : public MnApplication {

public:

  /// construct from FCNBase + std::vector for parameters and errors
  MnSimplex(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& err, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par,err), MnStrategy(stra)), fMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and covariance
  MnSimplex(const FCNBase& fcn, const std::vector<double>& par, const std::vector<double>& cov, unsigned int nrow, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov, nrow), MnStrategy(stra)), fMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + std::vector for parameters and MnUserCovariance
  MnSimplex(const FCNBase& fcn, const std::vector<double>& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), fMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + MnUserParameters
  MnSimplex(const FCNBase& fcn, const MnUserParameters& par, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par), MnStrategy(stra)), fMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + MnUserParameters + MnUserCovariance
  MnSimplex(const FCNBase& fcn, const MnUserParameters& par, const MnUserCovariance& cov, unsigned int stra = 1) : MnApplication(fcn, MnUserParameterState(par, cov), MnStrategy(stra)), fMinimizer(SimplexMinimizer()) {}

  /// construct from FCNBase + MnUserParameterState + MnStrategy
  MnSimplex(const FCNBase& fcn, const MnUserParameterState& par, const MnStrategy& str) : MnApplication(fcn, MnUserParameterState(par), str), fMinimizer(SimplexMinimizer()) {}

  MnSimplex(const MnSimplex& migr) : MnApplication(migr.Fcnbase(), migr.State(), migr.Strategy(), migr.NumOfCalls()), fMinimizer(migr.fMinimizer) {}  

  ~MnSimplex() {}

  const ModularFunctionMinimizer& Minimizer() const {return fMinimizer;}

private:

  SimplexMinimizer fMinimizer;

private:

  //forbidden assignment of migrad (const FCNBase& = )
  MnSimplex& operator=(const MnSimplex&) {return *this;}
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnSimplex
