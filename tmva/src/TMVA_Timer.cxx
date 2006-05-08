/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Timer                                                            *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
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

#include "TMVA_Timer.h"
#include "Riostream.h"

const TString BC_white  = "\033[1;37m" ;
const TString BC_red    = "\033[1;31m" ;
const TString BC_blue   = "\033[1;34m" ;
const TString BC__b0    = "\033[47m"   ;
const TString BC__b1    = "\033[1;42m" ;
const TString BC__f1    = "\033[33m"   ;
const TString EC__      = "\033[0m"    ;

const TString TMVA_Timer::fMethodName = "TMVA_Timer";
const Int_t   TMVA_Timer::fNbins      = 24;  

ClassImp(TMVA_Timer)

//_______________________________________________________________________
TMVA_Timer::TMVA_Timer( Bool_t colourfulOutput )
  : fNcounts        ( 0 ),
    fPrefix         ( TMVA_Timer::fMethodName ),
    fColourfulOutput( colourfulOutput )
{
  Reset();
}

//_______________________________________________________________________
TMVA_Timer::TMVA_Timer( Int_t ncounts, TString prefix, Bool_t colourfulOutput  )
  : fNcounts        ( ncounts ),
    fColourfulOutput( colourfulOutput )
{
  if (prefix == "") fPrefix = TMVA_Timer::fMethodName;
  else              fPrefix = prefix;

  Reset();
}

//_______________________________________________________________________
TMVA_Timer::~TMVA_Timer( void )
{}

void TMVA_Timer::Init( Int_t ncounts )
{
  fNcounts = ncounts;  
  Reset();
}

//_______________________________________________________________________
void TMVA_Timer::Reset( void )
{
  TStopwatch::Start( kTRUE );
}

//_______________________________________________________________________
Double_t TMVA_Timer::ElapsedSeconds( void ) 
{
  Double_t rt = TStopwatch::RealTime(); TStopwatch::Start( kFALSE );
  return rt;
}
//_______________________________________________________________________

TString TMVA_Timer::GetElapsedTime( Bool_t Scientific ) 
{
  return SecToText( ElapsedSeconds(), Scientific );
}

//_______________________________________________________________________
TString TMVA_Timer::GetLeftTime( Int_t icounts ) 
{
  Double_t leftTime = ( icounts <= 0 ? -1 :
			icounts > fNcounts ? -1 :
			Double_t(fNcounts - icounts)/Double_t(icounts)*ElapsedSeconds() );

  return SecToText( leftTime, kFALSE );
}

//_______________________________________________________________________
void TMVA_Timer::DrawProgressBar( Int_t icounts ) 
{
  // sanity check:
  if (icounts > fNcounts-1) icounts = fNcounts-1;
  if (icounts < 0          ) icounts = 0;
  Int_t ic = Int_t(Float_t(icounts)/Float_t(fNcounts)*fNbins);

  clog << "--- " << fPrefix << ": ";
  if (fColourfulOutput) clog << BC__b1 << BC__f1 << "[" << EC__;
  else                   clog << "[";
  for (Int_t i=0; i<ic; i++) {
    if (fColourfulOutput) clog << BC__b1 << BC__f1 << ">" << EC__; 
    else                   clog << ">";
  }
  for (Int_t i=ic+1; i<fNbins; i++) {
    if (fColourfulOutput) clog << BC__b1 << BC__f1 << "." << EC__; 
    else                   clog << ".";
  }
  if (fColourfulOutput) clog << BC__b1 << BC__f1 << "]" << EC__;
  else                   clog << "]" ;

  // timing information
  if (fColourfulOutput) {
    clog << EC__ << " " ;
    clog << "(" << BC_red << Int_t((100*(icounts+1))/Float_t(fNcounts)) << "%" << EC__
 	 << ", " 
 	 << "time left: "
 	 << this->GetLeftTime( icounts ) << EC__ << ") ";
  }
  else {
    clog << "] " ;
    clog << "(" << Int_t((100*(icounts+1))/Float_t(fNcounts)) << "%" 
 	 << ", " << "time left: " << this->GetLeftTime( icounts ) << ") ";
  }
  clog << "\r" << flush; 
}

//_______________________________________________________________________
TString TMVA_Timer::SecToText( Double_t seconds, Bool_t Scientific ) 
{
  TString out = "";
  if      (Scientific    ) out = Form( "%.3g sec", seconds );
  else if (seconds <  0  ) out = "unknown";
  else if (seconds <= 300) out = Form( "%i sec", Int_t(seconds) );
  else {
    if (seconds > 3600) {
      Int_t h = Int_t(seconds/3600);
      if (h <= 1) out = Form( "%i hr : ", h );
      else        out = Form( "%i hrs : ", h );
      
      seconds = Int_t(seconds)%3600;
    }
    Int_t m = Int_t(seconds/60);
    if (m <= 1) out += Form( "%i min", m );
    else        out += Form( "%i mins", m );
  }

  return (fColourfulOutput) ? BC_red + out + EC__ : out;
}

