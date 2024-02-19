// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSpline1                                                              *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::TSpline1
\ingroup TMVA
Linear interpolation of TGraph
*/

#include "TMVA/TSpline1.h"

#include "TGraph.h"
#include "TMath.h"

ClassImp(TMVA::TSpline1);

////////////////////////////////////////////////////////////////////////////////
/// constructor from TGraph pointer (not owned by TSpline1)
/// TSpline is a TNamed object

TMVA::TSpline1::TSpline1( const TString& title, const TGraph *theGraph )
   : N(theGraph->GetN()),
     X(theGraph->GetX(), theGraph->GetX() + N),
     Y(theGraph->GetY(), theGraph->GetY() + N)
{
   SetNameTitle( title, title );
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::TSpline1::~TSpline1( void ) {}

////////////////////////////////////////////////////////////////////////////////
/// returns linearly interpolated TGraph entry around x

Double_t TMVA::TSpline1::Eval( Double_t x ) const
{
   Int_t ibin = std::distance(X.begin(), TMath::BinarySearch(X.begin(), X.end(), x));
   // sanity checks
   if (ibin < 0 ) ibin = 0;
   if (ibin >= N) ibin = N - 1;

   Int_t nextbin = ibin;
   if ((x > X[ibin] && ibin != N-1) || ibin == 0)
      nextbin++;
   else
      nextbin--;

   // linear interpolation
   Double_t dx = X[ibin] - X[nextbin];
   Double_t dy = Y[ibin] - Y[nextbin];
   return Y[ibin] + (x - X[ibin]) * dy/dx;
}

////////////////////////////////////////////////////////////////////////////////
/// no coefficients to precompute

void TMVA::TSpline1::BuildCoeff( void )
{
}

////////////////////////////////////////////////////////////////////////////////
/// no knots

void TMVA::TSpline1::GetKnot( Int_t /* i*/, Double_t&  /*x*/, Double_t& /*y*/ ) const
{
}
