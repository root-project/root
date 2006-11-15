// @(#)root/tmva $Id: Timer.cxx,v 1.13 2006/10/15 22:34:22 andreas.hoecker Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::Timer                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Timing information for training and evaluation of MVA methods  
// 
// Usage:
//
//    TMVA::Timer timer( Nloops, "MyClassName" ); 
//    for (Int_t i=0; i<Nloops; i++) {
//      ... // some code
//
//      // now, print progress bar:
//      timer.DrawProgressBar( i );
//
//      // **OR** text output of left time (never both !)
//      fLogger << " time left: " << timer.GetLeftTime( i ) << Endl;
//
//    } 
//    fLogger << "MyClassName" << ": elapsed time: " << timer.GetElapsedTime() 
//            << Endl;    
//
// Remark: in batch mode, the progress bar is quite ugly; you may 
//         want to use the text output then
//_______________________________________________________________________

#include "TMVA/Timer.h"
#include "Riostream.h"

const TString BC_white  = "\033[1;37m" ;
const TString BC_red    = "\033[0;31m" ;
const TString BC_blue   = "\033[0;34m" ;
const TString BC__b0    = "\033[47m"   ;
const TString BC__b1    = "\033[1;42m" ;
const TString BC__f1    = "\033[33m"   ;
const TString EC__      = "\033[0m"    ;

const TString TMVA::Timer::fgClassName = "Timer";
const Int_t   TMVA::Timer::fgNbins     = 24;  

ClassImp(TMVA::Timer)
   ;

//_______________________________________________________________________
TMVA::Timer::Timer( Bool_t colourfulOutput )
   : fNcounts        ( 0 ),
     fPrefix         ( TMVA::Timer::fgClassName ),
     fColourfulOutput( colourfulOutput )
{
   // constructor
   fLogger = new MsgLogger( fPrefix.Data() );

   Reset();
}

//_______________________________________________________________________
TMVA::Timer::Timer( Int_t ncounts, TString prefix, Bool_t colourfulOutput  )
   : fNcounts        ( ncounts ),
     fColourfulOutput( colourfulOutput )
{
   // standard constructor: ncounts gives the total number of counts that 
   // the loop will iterate through. At each call of the timer, the current
   // number of counts is provided by the user, so that the timer can obtain
   // the due time from linearly interpolating the spent time.
   if (prefix == "") fPrefix = TMVA::Timer::fgClassName;
   else              fPrefix = prefix;

   fLogger = new MsgLogger( fPrefix.Data() );

   Reset();
}

//_______________________________________________________________________
TMVA::Timer::~Timer( void )
{
   // destructor
   delete fLogger;
}

void TMVA::Timer::Init( Int_t ncounts )
{
   // timer initialisation
   fNcounts = ncounts;  
   Reset();
}

//_______________________________________________________________________
void TMVA::Timer::Reset( void )
{
   // resets timer
   TStopwatch::Start( kTRUE );
}

//_______________________________________________________________________
Double_t TMVA::Timer::ElapsedSeconds( void ) 
{
   // computes elapsed tim in seconds
   Double_t rt = TStopwatch::RealTime(); TStopwatch::Start( kFALSE );
   return rt;
}
//_______________________________________________________________________

TString TMVA::Timer::GetElapsedTime( Bool_t Scientific ) 
{
   // returns pretty string with elaplsed time
   return SecToText( ElapsedSeconds(), Scientific );
}

//_______________________________________________________________________
TString TMVA::Timer::GetLeftTime( Int_t icounts ) 
{
   // returns pretty string with time left
   Double_t leftTime = ( icounts <= 0 ? -1 :
                         icounts > fNcounts ? -1 :
                         Double_t(fNcounts - icounts)/Double_t(icounts)*ElapsedSeconds() );

   return SecToText( leftTime, kFALSE );
}

//_______________________________________________________________________
void TMVA::Timer::DrawProgressBar( Int_t icounts ) 
{
   // draws progress bar in color or B&W
   // caution: 

   // sanity check:
   if (icounts > fNcounts-1) icounts = fNcounts-1;
   if (icounts < 0         ) icounts = 0;
   Int_t ic = Int_t(Float_t(icounts)/Float_t(fNcounts)*fgNbins);

   clog << fLogger->GetPrintedSource();
   if (fColourfulOutput) clog << BC__b1 << BC__f1 << "[" << EC__;
   else                  clog << "[";
   for (Int_t i=0; i<ic; i++) {
      if (fColourfulOutput) clog << BC__b1 << BC__f1 << ">" << EC__; 
      else                  clog << ">";
   }
   for (Int_t i=ic+1; i<fgNbins; i++) {
      if (fColourfulOutput) clog << BC__b1 << BC__f1 << "." << EC__; 
      else                  clog << ".";
   }
   if (fColourfulOutput) clog << BC__b1 << BC__f1 << "]" << EC__;
   else                  clog << "]" ;

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
TString TMVA::Timer::SecToText( Double_t seconds, Bool_t Scientific ) 
{
   // pretty string output
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

