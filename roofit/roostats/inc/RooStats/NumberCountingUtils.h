// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_NumberCountingUtils
#define RooStats_NumberCountingUtils

/*! \namespace NumberCountingUtils
    \brief RooStats standalone utilities

These are RooStats standalone utilities
that calculate the p-value or Z value (eg. significance in
1-sided Gaussian standard deviations) for a number counting experiment.
This is a hypothesis test between background only and signal-plus-background.
The background estimate has uncertainty derived from an auxiliary or sideband
measurement.

This is based on code and comments from Bob Cousins
and on the following papers:

  - Evaluation of three methods for calculating statistical significance when incorporating a
    systematic uncertainty into a test of the background-only hypothesis for a Poisson process<br>
    Authors: Robert D. Cousins, James T. Linnemann, Jordan Tucker<br>
    http://arxiv.org/abs/physics/0702156<br>
    NIM  A 595 (2008) 480--501<br>


  - Statistical Challenges for Searches for New Physics at the LHC<br>
    Authors: Kyle Cranmer<br>
    http://arxiv.org/abs/physics/0511028

  - Measures of Significance in HEP and Astrophysics<br>
    Authors: J. T. Linnemann<br>
    http://arxiv.org/abs/physics/0312059

The problem is treated in a fully frequentist fashion by
interpreting the relative background uncertainty as
being due to an auxiliary or sideband observation
that is also Poisson distributed with only background.
Finally, one considers the test as a ratio of Poisson means
where an interval is well known based on the conditioning on the total
number of events and the binomial distribution.

In short, this is an exact frequentist solution to the problem of
a main measurement x distributed as a Poisson around s+b and a sideband or
auxiliary measurement y distributed as a Poisson around tau*b.  Eg.

\f[ L(x,y|s,b,\tau) = Pois(x|s+b) Pois(y|\tau b) \f]

```
Naming conventions:
Exp = Expected
Obs = Observed
P   = p-value
Z   = Z-value or significance in sigma (one-sided convention)
```
*/

#include "RtypesCore.h"


namespace RooStats{

   namespace  NumberCountingUtils {


  /// Expected P-value for s=0 in a ratio of Poisson means.
  /// Here the background and its uncertainty are provided directly and
  /// assumed to be from the double Poisson counting setup described in the
  /// BinomialWithTau functions.
  /// Normally one would know tau directly, but here it is determined from
  /// the background uncertainty.  This is not strictly correct, but a useful
  /// approximation.
     double BinomialExpZ(double sExp, double bExp, double fractionalBUncertainty);

  /// See BinomialWithTauExpP
     double BinomialWithTauExpZ(double sExp, double bExp, double tau);

  /// See BinomialObsP
     double BinomialObsZ(double nObs, double bExp, double fractionalBUncertainty);

  /// See BinomialWithTauObsP
     double BinomialWithTauObsZ(double nObs, double bExp, double tau);

  /// See BinomialExpP
     double BinomialExpP(double sExp, double bExp, double fractionalBUncertainty);

  /// Expected P-value for s=0 in a ratio of Poisson means.
  /// Based on two expectations, a main measurement that might have signal
  /// and an auxiliary measurement for the background that is signal free.
  /// The expected background in the auxiliary measurement is a factor
  /// tau larger than in the main measurement.
     double BinomialWithTauExpP(double sExp, double bExp, double tau);

  /// P-value for s=0 in a ratio of Poisson means.
  /// Here the background and its uncertainty are provided directly and
  /// assumed to be from the double Poisson counting setup.
  /// Normally one would know tau directly, but here it is determined from
  /// the background uncertainty.  This is not strictly correct, but a useful
  /// approximation.
     double BinomialObsP(double nObs, double, double fractionalBUncertainty);

  /// P-value for s=0 in a ratio of Poisson means.
  /// Based on two observations, a main measurement that might have signal
  /// and an auxiliary measurement for the background that is signal free.
  /// The expected background in the auxiliary measurement is a factor
  /// tau larger than in the main measurement.
     double BinomialWithTauObsP(double nObs, double bExp, double tau);


   }
}

#endif
