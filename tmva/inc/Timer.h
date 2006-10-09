// @(#)root/tmva $Id: Timer.h,v 1.7 2006/08/30 22:20:00 andreas.hoecker Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Timer                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Timing information for methods training                                   *
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
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_Timer
#define ROOT_TMVA_Timer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Timer                                                                //
//                                                                      //
// Timing information for training and evaluation of MVA methods        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "time.h"
#include "TString.h"
#include "TStopwatch.h"

// ensure that clock_t is always defined
#if defined(__SUNPRO_CC) && defined(_XOPEN_SOURCE) && (_XOPEN_SOURCE - 0 == 500 )
#ifndef _CLOCK_T
#define _CLOCK_T
typedef long clock_t; // relative time in a specified resolution
#endif /* ifndef _CLOCK_T */

#endif // SUN and XOPENSOURCE=500

namespace TMVA {

   class Timer : public TStopwatch {
  
   public:
  
      Timer( Bool_t colourfulOutput = kTRUE );
      Timer( Int_t ncounts, TString prefix = "", Bool_t colourfulOutput = kTRUE );
      virtual ~Timer( void );
  
      void Init ( Int_t ncounts );
      void Reset( void );

      // when the "Scientific" flag set, time is returned with subdecimals
      // for algorithm timing measurement
      TString GetElapsedTime  ( Bool_t Scientific = kTRUE  );
      TString GetLeftTime     ( Int_t icounts              );
      void    DrawProgressBar ( Int_t icounts              );
                          
   private:

      Double_t  ElapsedSeconds( void             );
      TString   SecToText     ( Double_t, Bool_t );

      Int_t     fNcounts;               // reference number of "counts" 
      TString   fPrefix;                // prefix for outputs
      Bool_t    fColourfulOutput;       // flag for use of colors

      static const TString fgClassName; // used for output
      static const Int_t   fgNbins;     // number of bins in progress bar

      ClassDef(Timer,0) // Timing information for training and evaluation of MVA methods
         };

} // namespace

#endif
