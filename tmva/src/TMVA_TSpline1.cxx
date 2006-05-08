/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_TSpline1                                                         *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#include "TMVA_TSpline1.h"
#include "Riostream.h"

ClassImp(TMVA_TSpline1)

//_______________________________________________________________________
TMVA_TSpline1::TMVA_TSpline1( TString title, TGraph* theGraph )
  : fGraph( theGraph )
{
  // TSpline is a TNamed object
  SetNameTitle( title, title );  
}

//_______________________________________________________________________
TMVA_TSpline1::~TMVA_TSpline1( void )
{
  if (NULL != fGraph) delete fGraph;
}

//_______________________________________________________________________
Double_t TMVA_TSpline1::Eval( Double_t x ) const
{  
  
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

  Double_t Dx = fGraph->GetX()[ibin] - fGraph->GetX()[nextbin];
  Double_t Dy = fGraph->GetY()[ibin] - fGraph->GetY()[nextbin];
  return fGraph->GetY()[ibin] + (x - fGraph->GetX()[ibin]) * Dy/Dx;
}

//_______________________________________________________________________
void TMVA_TSpline1::BuildCoeff( void )
{}

//_______________________________________________________________________
void TMVA_TSpline1::GetKnot( Int_t /* i*/, Double_t&  /*x*/, Double_t& /*y*/ ) const
{}

