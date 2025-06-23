// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSpline2                                                              *
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

/*! \class TMVA::TSpline2
\ingroup TMVA
Quadratic interpolation of TGraph
*/

#include "TMVA/TSpline2.h"

#include "TGraph.h"
#include "TMath.h"
#include "TSpline.h"

ClassImp(TMVA::TSpline2);

////////////////////////////////////////////////////////////////////////////////
/// constructor from TGraph pointer (not owned by TSpline2)
/// TSpline is a TNamed object

TMVA::TSpline2::TSpline2(const TString &title, const TGraph *theGraph)
: fX(theGraph->GetX(), theGraph->GetX() + theGraph->GetN()),
  fY(theGraph->GetY(), theGraph->GetY() + theGraph->GetN())
{
   SetNameTitle( title, title );
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::TSpline2::~TSpline2(void) {}

////////////////////////////////////////////////////////////////////////////////
/// returns quadratically interpolated TGraph entry around x

Double_t TMVA::TSpline2::Eval( const Double_t x ) const
{
   Double_t retval=0;
   Int_t N = fX.size();
   Int_t ibin = std::distance(std::lower_bound(fX.rbegin(), fX.rend(), x, std::greater{}), fX.rend()) - 1;

   // sanity checks
   if (ibin < 0 ) ibin = 0;
   if (ibin >= N) ibin = N - 1;

   Float_t dx = 0; // should be zero
   if (N < 3) { // if the graph does not have enough points
      Warning("Eval", "Graph has less than 3 points, returning value of the closest");
      retval = fY[ibin];
   } else if (ibin == 0) {

      retval = Quadrax(x,
                       fX[ibin] + dx,
                       fX[ibin + 1] + dx,
                       fX[ibin + 2] + dx,
                       fY[ibin],
                       fY[ibin + 1],
                       fY[ibin + 2]);

   } else if (ibin >= (N - 2)) {
      ibin = N - 1; // always fixed to last bin

      retval = Quadrax(x,
                       fX[ibin - 2] + dx,
                       fX[ibin - 1] + dx,
                       fX[ibin] + dx,
                       fY[ibin - 2],
                       fY[ibin - 1],
                       fY[ibin]);
   } else {

      retval = ( Quadrax( x,
                          fX[ibin-1] + dx,
                          fX[ibin]   + dx,
                          fX[ibin+1] + dx,
                          fY[ibin-1],
                          fY[ibin],
                          fY[ibin+1])
                 +
                 Quadrax( x,
                          fX[ibin] + dx,
                          fX[ibin+1]  + dx,
                          fX[ibin+2]  + dx,
                          fY[ibin],
                          fY[ibin+1],
                          fY[ibin+2]) )*0.5;
   }

   return retval;
}

////////////////////////////////////////////////////////////////////////////////
/// no coefficients to precompute

void TMVA::TSpline2::BuildCoeff( void )
{
}

////////////////////////////////////////////////////////////////////////////////
/// no knots

void TMVA::TSpline2::GetKnot( Int_t  /*i*/, Double_t& /*x*/, Double_t& /*y*/ ) const
{
}

////////////////////////////////////////////////////////////////////////////////
/// quadratic interpolation
/// Revised and checked by Francois Nov, 16th, 2000
/// Note the beautiful non-spontaneous symmetry breaking ...
/// It was checked that the old routine gave exactly the same answers.
///

Double_t TMVA::TSpline2::Quadrax( const Float_t dm,const Float_t dm1,const Float_t dm2,const Float_t dm3,
                                  const Float_t cos1, const Float_t cos2, const Float_t cos3 ) const
{
   Float_t a = cos1*(dm2-dm3) + cos2*(dm3-dm1) + cos3*(dm1-dm2);
   Float_t b = cos1*(dm2*dm2-dm3*dm3) + cos2*(dm3*dm3-dm1*dm1) + cos3*(dm1*dm1-dm2*dm2);
   Float_t c = cos1*(dm2-dm3)*dm2*dm3 + cos2*(dm3-dm1)*dm3*dm1 + cos3*(dm1-dm2)*dm1*dm2;

   Float_t denom = (dm2-dm3)*(dm3-dm1)*(dm1-dm2);

   return (denom != 0.0) ? (-a*dm*dm+b*dm-c)/denom : 0.0;
}


