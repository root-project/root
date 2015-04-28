// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSpline2                                                              *
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
// Quadratic interpolation of TGraph
//_______________________________________________________________________

#include "TMath.h"

#include "TMVA/TSpline2.h"

ClassImp(TMVA::TSpline2)

//_______________________________________________________________________
TMVA::TSpline2::TSpline2( const TString& title, TGraph* theGraph )
   : fGraph( theGraph ) // not owned by TSpline2
{
   // constructor from TGraph
   // TSpline is a TNamed object
   SetNameTitle( title, title );
}

//_______________________________________________________________________
TMVA::TSpline2::~TSpline2( void )
{
   // destructor
   if (fGraph) delete fGraph; // ROOT's spline classes also own the TGraph
}

//_______________________________________________________________________
Double_t TMVA::TSpline2::Eval( const Double_t x ) const
{  
   // returns quadratically interpolated TGraph entry around x
   Double_t retval=0;
  
   Int_t ibin = TMath::BinarySearch( fGraph->GetN(),
                                     fGraph->GetX(),
                                     x );

   // sanity checks
   if (ibin < 0               ) ibin = 0;
   if (ibin >= fGraph->GetN()) ibin =  fGraph->GetN() - 1;
  
   Float_t dx = 0; // should be zero
  
   if (ibin == 0 ) {
    
      retval = Quadrax(  x,
                         fGraph->GetX()[ibin]   + dx,
                         fGraph->GetX()[ibin+1] + dx,
                         fGraph->GetX()[ibin+2] + dx,
                         fGraph->GetY()[ibin],
                         fGraph->GetY()[ibin+1],
                         fGraph->GetY()[ibin+2]);
    
   }
   else if (ibin >= (fGraph->GetN()-2)) {
      ibin = fGraph->GetN() - 1; // always fixed to last bin

      retval = Quadrax( x,
                        fGraph->GetX()[ibin-2] + dx,
                        fGraph->GetX()[ibin-1] + dx,
                        fGraph->GetX()[ibin]   + dx,
                        fGraph->GetY()[ibin-2],
                        fGraph->GetY()[ibin-1],
                        fGraph->GetY()[ibin]);
   } 
   else {  
    
      retval = ( Quadrax( x, 
                          fGraph->GetX()[ibin-1] + dx,
                          fGraph->GetX()[ibin]   + dx,
                          fGraph->GetX()[ibin+1] + dx,
                          fGraph->GetY()[ibin-1],
                          fGraph->GetY()[ibin],
                          fGraph->GetY()[ibin+1])
                 + 
                 Quadrax( x, fGraph->GetX()[ibin] + dx,
                          fGraph->GetX()[ibin+1]  + dx,
                          fGraph->GetX()[ibin+2]  + dx,
                          fGraph->GetY()[ibin],
                          fGraph->GetY()[ibin+1],
                          fGraph->GetY()[ibin+2]) )*0.5;
   }

   return retval;
}

//_______________________________________________________________________
void TMVA::TSpline2::BuildCoeff( void )
{
   // no coefficients to precompute
}

//_______________________________________________________________________
void TMVA::TSpline2::GetKnot( Int_t  /*i*/, Double_t& /*x*/, Double_t& /*y*/ ) const
{
   // no knots
}

//_______________________________________________________________________
Double_t TMVA::TSpline2::Quadrax( const Float_t dm,const Float_t dm1,const Float_t dm2,const Float_t dm3,
                                  const Float_t cos1, const Float_t cos2, const Float_t cos3 ) const
{  
   // quadratic interpolation
   // Revised and checked by Francois Nov, 16th, 2000
   // Note the beautiful non-spontaneous symmetry breaking ...
   // It was checked that the old routine gave exactly the same answers.
   //   
   Float_t a = cos1*(dm2-dm3) + cos2*(dm3-dm1) + cos3*(dm1-dm2);
   Float_t b = cos1*(dm2*dm2-dm3*dm3) + cos2*(dm3*dm3-dm1*dm1) + cos3*(dm1*dm1-dm2*dm2);
   Float_t c = cos1*(dm2-dm3)*dm2*dm3 + cos2*(dm3-dm1)*dm3*dm1 + cos3*(dm1-dm2)*dm1*dm2;

   Float_t denom = (dm2-dm3)*(dm3-dm1)*(dm1-dm2);
  
   return (denom != 0.0) ? (-a*dm*dm+b*dm-c)/denom : 0.0;
}


