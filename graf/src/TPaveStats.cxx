// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   15/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "TPaveStats.h"
#include "TStyle.h"
#include "TFile.h"

ClassImp(TPaveStats)

//______________________________________________________________________________
//  A PaveStats is a PaveText to draw histogram statistics
// The type of information printed in the histogram statistics box
//  can be selected via gStyle->SetOptStat(mode).
//  or by editing an existing TPaveStats object via TPaveStats::SetOptStat(mode).
//  The parameter mode can be = ourmen  (default = 001111)
//    n = 1;  name of histogram is printed
//    e = 1;  number of entries printed
//    m = 1;  mean value printed
//    r = 1;  rms printed
//    u = 1;  number of underflows printed
//    o = 1;  number of overflows printed
//  Example: gStyle->SetOptStat(11);
//           print only name of histogram and number of entries.
//
// The type of information about fit parameters printed in the histogram
// statistics box can be selected via the parameter mode.
//  The parameter mode can be = pcev  (default = 0111)
//    v = 1;  print name/values of parameters
//    e = 1;  print errors (if e=1, v must be 1)
//    c = 1;  print Chisquare/Number of degress of freedom
//    p = 1;  print Probability
//  Example: gStyle->SetOptFit(1011);
//        or this->SetOptFit(1011);
//           print fit probability, parameter names/values and errors.
//

//______________________________________________________________________________
TPaveStats::TPaveStats(): TPaveText()
{
//*-*-*-*-*-*-*-*-*-*-*pavetext default constructor*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =============================

}

//______________________________________________________________________________
TPaveStats::TPaveStats(Coord_t x1, Coord_t y1,Coord_t x2, Coord_t  y2, Option_t *option)
           :TPaveText(x1,y1,x2,y2,option)
{
//*-*-*-*-*-*-*-*-*-*-*pavetext normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
//

   fOptFit  = gStyle->GetOptFit();
   fOptStat = gStyle->GetOptStat();
   SetFitFormat(gStyle->GetFitFormat());
   SetStatFormat(gStyle->GetStatFormat());
}

//______________________________________________________________________________
TPaveStats::~TPaveStats()
{
//*-*-*-*-*-*-*-*-*-*-*pavetext default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================
}

//______________________________________________________________________________
void TPaveStats::SaveStyle()
{
//  Save This TPaveStats options in current style
//
   gStyle->SetOptFit(fOptFit);
   gStyle->SetOptStat(fOptStat);
   gStyle->SetFitFormat(fFitFormat.Data());
   gStyle->SetStatFormat(fStatFormat.Data());
}

//______________________________________________________________________________
void TPaveStats::SetFitFormat(const char *form)
{
   // Change (i.e. set) the format for printing fit parameters in statistics box

   fFitFormat = form;
}

//______________________________________________________________________________
void TPaveStats::SetStatFormat(const char *form)
{
   // Change (i.e. set) the format for printing statistics

   fStatFormat = form;
}


//______________________________________________________________________________
void TPaveStats::Streamer(TBuffer &R__b)
{
   // Stream an object of class TPaveStats.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      TPaveText::Streamer(R__b);
      R__b >> fOptFit;
      R__b >> fOptStat;
      if (R__v > 1 || (gFile && gFile->GetVersion() == 22304)) {
         fFitFormat.Streamer(R__b);
         fStatFormat.Streamer(R__b);
      } else {
         SetFitFormat();
         SetStatFormat();
      }
      R__b.CheckByteCount(R__s, R__c, TPaveStats::IsA());
   } else {
      R__c = R__b.WriteVersion(TPaveStats::IsA(), kTRUE);
      TPaveText::Streamer(R__b);
      R__b << fOptFit;
      R__b << fOptStat;
      fFitFormat.Streamer(R__b);
      fStatFormat.Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}
