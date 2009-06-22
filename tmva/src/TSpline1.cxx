// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSpline1                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Linear interpolation of TGraph
//_______________________________________________________________________

#include "TMath.h"

#include "TMVA/TSpline1.h"

ClassImp(TMVA::TSpline1)

//_______________________________________________________________________
TMVA::TSpline1::TSpline1( const TString& title, TGraph* theGraph )
   : fGraph( theGraph )
{
   // constructor from TGraph
   // TSpline is a TNamed object
   SetNameTitle( title, title );  
}

//_______________________________________________________________________
TMVA::TSpline1::~TSpline1( void )
{
   // destructor
   if (fGraph) delete fGraph; // ROOT's spline classes also own the TGraph
}

//_______________________________________________________________________
Double_t TMVA::TSpline1::Eval( Double_t x ) const
{  
   // returns linearly interpolated TGraph entry around x
   Int_t ibin = TMath::BinarySearch( fGraph->GetN(),
                                     fGraph->GetX(),
                                     x );
   Int_t nbin = fGraph->GetN();

   // sanity checks
   if (ibin < 0    ) ibin = 0;
   if (ibin >= nbin) ibin = nbin - 1;

   Int_t nextbin = ibin;
   if ((x > fGraph->GetX()[ibin] && ibin != nbin-1) || ibin == 0) 
      nextbin++;
   else
      nextbin--;  

   // linear interpolation
   Double_t dx = fGraph->GetX()[ibin] - fGraph->GetX()[nextbin];
   Double_t dy = fGraph->GetY()[ibin] - fGraph->GetY()[nextbin];
   return fGraph->GetY()[ibin] + (x - fGraph->GetX()[ibin]) * dy/dx;
}

//_______________________________________________________________________
void TMVA::TSpline1::BuildCoeff( void )
{
   // no coefficients to precompute
}

//_______________________________________________________________________
void TMVA::TSpline1::GetKnot( Int_t /* i*/, Double_t&  /*x*/, Double_t& /*y*/ ) const
{
   // no knots
}

