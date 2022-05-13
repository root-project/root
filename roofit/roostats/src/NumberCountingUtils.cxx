// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "RooStats/NumberCountingUtils.h"

#include "RooStats/RooStatsUtils.h"

// // Without this macro the THtml doc  can not be generated
// #if !defined(R__ALPHA) && !defined(R__SOLARIS) && !defined(R__ACC) && !defined(R__FBSD)
// NamespaceImp(RooStats)
// //NamespaceImp(NumberCountingUtils)
// #endif

//using namespace RooStats;

double RooStats::NumberCountingUtils::BinomialExpP(double signalExp, double backgroundExp, double relativeBkgUncert){
  // Expected P-value for s=0 in a ratio of Poisson means.
  // Here the background and its uncertainty are provided directly and
  // assumed to be from the double Poisson counting setup described in the
  // BinomialWithTau functions.
  // Normally one would know tau directly, but here it is determiend from
  // the background uncertainty.  This is not strictly correct, but a useful
  // approximation.


  //SIDE BAND EXAMPLE
  //See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
  //150 total events in signalExp region, 100 in sideband of equal size
  double mainInf = signalExp+backgroundExp;  //Given
  double tau = 1./backgroundExp/(relativeBkgUncert*relativeBkgUncert);
  double auxiliaryInf = backgroundExp*tau;  //Given

  double P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainInf,auxiliaryInf+1);
  return P_Bi;

/*
Now, if instead the mean background level b in the signal region is
specified, along with Gaussian rms sigb, then one can fake a Poisson
sideband (see Linnemann, p. 35, converted to Cranmer's notation) by
letting tau = b/(sigb*sigb) and y = b*tau.  Thus, for example, if one
has x=150 and b = 100 +/- 10, one then derives tau and y.  Then one
has the same two lines of ROOT calling BetaIncomplete and ErfInverse.
Since I chose these numbers to revert to the previous example, we get
the same answer:
*/
/*
//GAUSSIAN FAKED AS POISSON EXAMPLE
x = 150.    //Given
b = 100.    //Given
sigb = 10.  //Given
tau = b/(sigb*sigb)
y = tau*b
Z_Bi = TMath::BetaIncomplete(1./(1.+tau),x,y+1)
S = sqrt(2)*TMath::ErfInverse(1 - 2*Z_Bi)

*/

}


double RooStats::NumberCountingUtils::BinomialWithTauExpP(double signalExp, double backgroundExp, double tau){
  // Expected P-value for s=0 in a ratio of Poisson means.
  // Based on two expectations, a main measurement that might have signal
  // and an auxiliarly measurement for the background that is signal free.
  // The expected background in the auxiliary measurement is a factor
  // tau larger than in the main measurement.

  double mainInf = signalExp+backgroundExp;  //Given
  double auxiliaryInf = backgroundExp*tau;  //Given

  double P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainInf,auxiliaryInf+1);

  return P_Bi;

}

double RooStats::NumberCountingUtils::BinomialObsP(double mainObs, double backgroundObs, double relativeBkgUncert){
  // P-value for s=0 in a ratio of Poisson means.
  // Here the background and its uncertainty are provided directly and
  // assumed to be from the double Poisson counting setup.
  // Normally one would know tau directly, but here it is determiend from
  // the background uncertainty.  This is not strictly correct, but a useful
  // approximation.

  double tau = 1./backgroundObs/(relativeBkgUncert*relativeBkgUncert);
  double auxiliaryInf = backgroundObs*tau;  //Given


  //SIDE BAND EXAMPLE
  //See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
  double P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainObs,auxiliaryInf+1);

  return P_Bi;

}


double RooStats::NumberCountingUtils::BinomialWithTauObsP(double mainObs, double auxiliaryObs, double tau){
  // P-value for s=0 in a ratio of Poisson means.
  // Based on two observations, a main measurement that might have signal
  // and an auxiliarly measurement for the background that is signal free.
  // The expected background in the auxiliary measurement is a factor
  // tau larger than in the main measurement.

  //SIDE BAND EXAMPLE
  //See Eqn. (19) of Cranmer and pp. 36-37 of Linnemann.
  double P_Bi = TMath::BetaIncomplete(1./(1.+tau),mainObs,auxiliaryObs+1);

  return P_Bi;

}

double RooStats::NumberCountingUtils::BinomialExpZ(double signalExp, double backgroundExp, double relativeBkgUncert) {
  // See BinomialExpP
  return RooStats::PValueToSignificance( BinomialExpP(signalExp,backgroundExp,relativeBkgUncert) ) ;
  }

double RooStats::NumberCountingUtils::BinomialWithTauExpZ(double signalExp, double backgroundExp, double tau){
  // See BinomialWithTauExpP
  return RooStats::PValueToSignificance( BinomialWithTauExpP(signalExp,backgroundExp,tau) ) ;
}


double RooStats::NumberCountingUtils::BinomialObsZ(double mainObs, double backgroundObs, double relativeBkgUncert){
  // See BinomialObsP
  return RooStats::PValueToSignificance( BinomialObsP(mainObs,backgroundObs,relativeBkgUncert) ) ;
}

double RooStats::NumberCountingUtils::BinomialWithTauObsZ(double mainObs, double auxiliaryObs, double tau){
  // See BinomialWithTauObsP
  return RooStats::PValueToSignificance( BinomialWithTauObsP(mainObs,auxiliaryObs,tau) ) ;
}
