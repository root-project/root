// @(#)root/physics:$Name:$:$Id:$
// Author: Adrian Bevan   10/02/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////
// TFeldmanCousins
//
// class to calculate the CL upper limit using
// the Feldman-Cousins method as described in PRD V57 #7, p3873-3889
//
// The default confidence interval calvculated using this method is 90%
// This is set either by having a default the constructor, or using the
// appropriate fraction when instantiating an object of this class (e.g. 0.9)
//
// The simple extension to a gaussian resolution function bounded at zero
// has not been addressed as yet -> `time is of the essence' as they write
// on the wall of the maze in that classic game ...
//
// Author: Adrian Bevan, Liverpool University
//
// Copyright Liverpool University 2001       bevan@slac.stanford.edu
///////////////////////////////////////////////////////////////////////////

#include "TFeldmanCousins.h"

ClassImp(TFeldmanCousins)

//______________________________________________________________________________
TFeldmanCousins::TFeldmanCousins(TString options)
{
  fCL          = 0.9;
  fUpperLimit  = 0.0;
  fLowerLimit  = 0.0;
  fNobserved   = 0.0;
  fNbackground = 0.0;
  options.ToLower();
  if (options.Contains("q")) fQUICK = 1;
  else                       fQUICK = 0;
}


//______________________________________________________________________________
TFeldmanCousins::TFeldmanCousins(Double_t newFC, TString options)
{
  fCL      = newFC;
  options.ToLower();
  if(options.Contains("q")) fQUICK = 1;
  else                      fQUICK = 0;
}


//______________________________________________________________________________
TFeldmanCousins::~TFeldmanCousins()
{
}


//______________________________________________________________________________
Double_t TFeldmanCousins::CalculateLowerLimit(Double_t Nobserved, Double_t Nbackground)
{
////////////////////////////////////////////////////////////////////////////////////////////
// given Nobserved and Nbackground, try different values of mu that give lower limits that//
// are consistent with Nobserved.  The closed interval (plus any stragglers) corresponds  //
// to the F&C interval                                                                    //
////////////////////////////////////////////////////////////////////////////////////////////

  CalculateUpperLimit(Nobserved, Nbackground);
  return fLowerLimit;
}


//______________________________________________________________________________
Double_t TFeldmanCousins::CalculateUpperLimit(Double_t Nobserved, Double_t Nbackground)
{
////////////////////////////////////////////////////////////////////////////////////////////
// given Nobserved and Nbackground, try different values of mu that give upper limits that//
// are consistent with Nobserved.  The closed interval (plus any stragglers) corresponds  //
// to the F&C interval                                                                    //
////////////////////////////////////////////////////////////////////////////////////////////

  fNobserved   = Nobserved;
  fNbackground = Nbackground;

  Double_t mu = 0.0;

  // for each mu construct the ranked table of probabilities and test the
  // observed number of events with the upper limit
  Double_t min = -999.0;
  Double_t max = 0;
  Int_t iLower = 0;

  Int_t i;
  for(i = 0; i <= nNuSteps; i++) {
    mu = MUMIN + (Double_t)i*MUSTEP;
    Int_t goodChoice = FindLimitsFromTable( mu );
    if( goodChoice ) {
      min = mu;
      iLower = i;
      break;
    }
  }

  //==================================================================
  // For quicker evaluation, assume that you get the same results when
  // you expect the uppper limit to be > Nobserved-Nbackground.
  // This is certainly true for all of the published tables in the PRD
  // and is a reasonable assumption in any case.
  //==================================================================

  Double_t quickJump = 0.0;
  if (fQUICK)          quickJump = Nobserved-Nbackground-MUMIN;
  if (quickJump < 0.0) quickJump = 0.0;

  for(i = iLower+1; i <= nNuSteps; i++) {
    mu = MUMIN + (Double_t)i*MUSTEP + quickJump;
    Int_t goodChoice = FindLimitsFromTable( mu );
    if( !goodChoice ) {
      max = mu;
      break;
    }
  }

  fUpperLimit = max;
  fLowerLimit = min;

  return max;
}


//______________________________________________________________________________
Int_t TFeldmanCousins::FindLimitsFromTable( Double_t mu )
{
///////////////////////////////////////////////////////////////////
// calculate the probability table for a given mu for n = 0, NMAX//
// and return 1 if the number of observed events is consistent   //
// with the CL bad                                               //
///////////////////////////////////////////////////////////////////

  Double_t P[NMAX];      //the array of probabilities in the interval MUMIN-MUMAX
  Double_t R[NMAX];      //the ratio of likliehoods = P(Mu|Nobserved)/P(MuBest|Nobserved)
  Int_t    rank[NMAX];   //the ranked array corresponding to R (largest first)
  Double_t MuBest[NMAX];
  Double_t ProbMuBest[NMAX];

  //calculate P(i | mu) and P(i | mu)/P(i | mubest)
  Int_t i;
  for(i = 0; i < NMAX; i++) {
    MuBest[i] = (Double_t)(i - fNbackground);
    if(MuBest[i]<0.0) MuBest[i] = 0.0;
    ProbMuBest[i] = Prob(i, MuBest[i],  fNbackground);
    P[i]          = Prob(i, mu,  fNbackground);
    if(ProbMuBest[i] == 0.0) R[i] = 0.0;
    else                     R[i] = P[i]/ProbMuBest[i];
  }

  //rank the likelihood ratio
  TMath::BubbleHigh(NMAX, R, rank);

  //search through the probability table and get the i for the CL
  Double_t sum = 0.0;
  Int_t iMax = rank[0];
  Int_t iMin = rank[0];
  for(i = 0; i < NMAX; i++) {
    sum += P[rank[i]];
    if(iMax < rank[i]) iMax = rank[i];
    if(iMin > rank[i]) iMin = rank[i];
    if(sum >= fCL) break;
  }

  if((fNobserved <= iMax) && (fNobserved >= iMin)) return 1;
  else return 0;
}


//______________________________________________________________________________
Double_t TFeldmanCousins::Prob(Int_t N, Double_t mu, Double_t B)
{
////////////////////////////////////////////////
// calculate the poissonian probability for   //
// a mean of mu+B events with a variance of N //
////////////////////////////////////////////////

  //calculate the factorial
  Double_t    factorial = 1.0;
  if (N == 2) factorial = 2.0;
  else if (N > 2) {
    for (Int_t ifact = N; ifact>=2; ifact--) {
      factorial = (Double_t)ifact * factorial;
    }
  }

  Double_t sum = mu+B;
  Double_t power;
  if      (N==1) power = sum;
  else if (N==2) power = sum*sum;
  else if (N==3) power = sum*sum*sum;
  else if (N==4) power = sum*sum*sum*sum;
  else           power = pow(sum, N);

  return ( power * exp(-sum) / factorial );
}
