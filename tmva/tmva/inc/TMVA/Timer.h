// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

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
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
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

#ifndef ROOT_time
#include "time.h"
#endif
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

   class MsgLogger;

   class Timer : public TStopwatch {

   public:

      Timer( const char* prefix = "", Bool_t colourfulOutput = kTRUE );
      Timer( Int_t ncounts, const char* prefix = "", Bool_t colourfulOutput = kTRUE );
      virtual ~Timer( void );

      void Init ( Int_t ncounts );
      void Reset( void );

      // when the "Scientific" flag set, time is returned with sub-decimals
      // for algorithm timing measurement
      TString   GetElapsedTime ( Bool_t Scientific = kTRUE  );
      Double_t  ElapsedSeconds ( void );
      TString   GetLeftTime     ( Int_t icounts );
      void      DrawProgressBar( Int_t, const TString& comment = "" );
      void      DrawProgressBar( TString );
      void      DrawProgressBar( void );

   private:

      TString   SecToText     ( Double_t, Bool_t ) const;

      Int_t     fNcounts;               ///< reference number of "counts"
      TString   fPrefix;                ///< prefix for outputs
      Bool_t    fColourfulOutput;       ///< flag for use of colors

      // Save state of previos progress
      Int_t     fPreviousProgress;
      TString   fPreviousTimeEstimate;
      Bool_t    fOutputToFile;

      Int_t     fProgressBarStringLength;

      static const TString fgClassName; ///< used for output
      static const Int_t   fgNbins;     ///< number of bins in progress bar

      mutable MsgLogger*   fLogger;     ///< the output logger
      MsgLogger& Log() const { return *fLogger; }

      ClassDef(Timer,0); // Timing information for training and evaluation of MVA methods
   };

} // namespace

#endif
