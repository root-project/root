// @(#)root/graf:$Name:  $:$Id: TPaveStats.cxx,v 1.9 2002/03/16 08:52:43 brun Exp $
// Author: Rene Brun   15/03/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "Riostream.h"
#include "TPaveStats.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TFile.h"
#include "TClass.h"
#include "TLatex.h"

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
   // TPaveStats default constructor
}

//______________________________________________________________________________
TPaveStats::TPaveStats(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Option_t *option)
           :TPaveText(x1,y1,x2,y2,option)
{
   // TPaveStats normal constructor

   fOptFit  = gStyle->GetOptFit();
   fOptStat = gStyle->GetOptStat();
   SetFitFormat(gStyle->GetFitFormat());
   SetStatFormat(gStyle->GetStatFormat());
}

//______________________________________________________________________________
TPaveStats::~TPaveStats()
{
   // TPaveStats default destructor
}

//______________________________________________________________________________
void TPaveStats::SaveStyle()
{
   //  Save This TPaveStats options in current style

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
void TPaveStats::Paint(Option_t *option)
{
   TPave::ConvertNDCtoPad();
   TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);

   if (!fLines) return;
   Double_t dx = fX2 - fX1;
   Double_t textsize = GetTextSize();
   Int_t nlines = GetSize();
   if (nlines == 0) nlines = 5;

   // Evaluate text size as a function of the number of lines
   Double_t y1       = gPad->GetY1();
   Double_t y2       = gPad->GetY2();
   Float_t margin    = fMargin*(fX2-fX1);
   Double_t yspace   = (fY2 - fY1)/Double_t(nlines);
   Double_t textsave = textsize;
   TObject *line;
   TLatex *latex, *latex_tok;
   TIter next(fLines);
   Double_t longest = 0;
   Double_t w, wtok[2];
   char *st, *sl;
   if (textsize == 0)  {
      textsize = 0.85*yspace/(y2 - y1);
      wtok[0] = 0; wtok[1] = 0;
      while ((line = (TObject*) next())) {
	 if (line->IsA() == TLatex::Class()) {
            latex = (TLatex*)line;
            sl = new char[strlen(latex->GetTitle())+1];
            strcpy(sl, latex->GetTitle());
            if (strpbrk(sl, "=") !=0) {
               st = strtok(sl, "=");
               Int_t itok = 0;
               while ( st !=0 ) {
                  latex_tok = new TLatex(0.,0.,st);
                  latex_tok->SetTextSize(textsize);
                  w = latex_tok->GetXsize();
                  if (w > wtok[itok]) wtok[itok] = w;
                  st = strtok(0, "=");
                  ++itok;
                  delete latex_tok;
               }
            }
         }
      }
      longest = wtok[0]+wtok[1]+2.*margin;
      if (longest > 0.98*dx) textsize *= 0.98*dx/longest;
      SetTextSize(textsize);
   }
   Double_t ytext = fY2 + 0.5*yspace;
   Double_t xtext = 0;

   // Iterate over all lines
   // Copy pavetext attributes to line attributes if line attributes not set
   next.Reset();
   while ((line = (TObject*) next())) {
      if (line->IsA() == TLatex::Class()) {
         latex = (TLatex*)line;
         ytext -= yspace;
         Double_t xl    = latex->GetX();
         Double_t yl    = latex->GetY();
         Short_t talign = latex->GetTextAlign();
         Color_t tcolor = latex->GetTextColor();
         Style_t tfont  = latex->GetTextFont();
         Size_t  tsize  = latex->GetTextSize();
         if (tcolor == 0) latex->SetTextColor(GetTextColor());
         if (tfont  == 0) latex->SetTextFont(GetTextFont());
         if (tsize  == 0) latex->SetTextSize(GetTextSize());

         sl = new char[strlen(latex->GetTitle())+1];
         strcpy(sl, latex->GetTitle());
	 // Draw all the histogram except the 2D under/overflow
         if (strpbrk(sl, "=") !=0) {
           st = strtok(sl, "=");
           Int_t halign = 12;
           while ( st !=0 ) {
              latex->SetTextAlign(halign);
              if (halign == 12) xtext = fX1 + margin;
              if (halign == 32) {
                 xtext = fX2 - margin;
		 // Clean trailing blanks in case of right alignment.
                 char *stc;
		 stc=st+strlen(st)-1;
		 while (*stc == ' ') {
                    *stc = '\0';
                    --stc;
                 }
              }
              latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                            latex->GetTextSize(),
                                            st);
              st = strtok(0, "=");
              halign = 32;
           }
	 // Draw the 2D under/overflow
         } else if (strpbrk(sl, "|") !=0) {
           Double_t Yline1 = ytext+yspace/2.;
           Double_t Yline2 = ytext-yspace/2.;
           Double_t Xline1 = (fX2-fX1)/3+fX1;
           Double_t Xline2 = 2*(fX2-fX1)/3+fX1;
           gPad->PaintLine(fX1,Yline1,fX2,Yline1);
           gPad->PaintLine(Xline1,Yline1,Xline1,Yline2);
           gPad->PaintLine(Xline2,Yline1,Xline2,Yline2);
           st = strtok(sl, "|");
	   Int_t Index = 0;
           while ( st !=0 ) {
              latex->SetTextAlign(22);
	      if (Index == 0) xtext = 0.5*(fX1+Xline1);
	      if (Index == 1) xtext = 0.5*(fX1+fX2);
	      if (Index == 2) xtext = 0.5*(Xline2+fX2);
              latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                            latex->GetTextSize(),
                                            st);
              Index++;
              st = strtok(0, "|");
           }
	 // Draw the histogram identifier
         } else {
           latex->SetTextAlign(22);
           xtext = 0.5*(fX1+fX2);
           latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                         latex->GetTextSize(),
                                         sl);
           gPad->PaintLine(fX1,fY2-yspace,fX2,fY2-yspace);
         }
         delete [] sl;

         latex->SetTextAlign(talign);
         latex->SetTextColor(tcolor);
         latex->SetTextFont(tfont);
         latex->SetTextSize(tsize);
         latex->SetX(xl);  //paintlatex modifies fX and fY
         latex->SetY(yl);
      }
   }
   SetTextSize(textsave);
}


//______________________________________________________________________________
void TPaveStats::Streamer(TBuffer &R__b)
{
   // Stream an object of class TPaveStats.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         TPaveStats::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TPaveText::Streamer(R__b);
      R__b >> fOptFit;
      R__b >> fOptStat;
      TFile *file = (TFile*)R__b.GetParent();
      if (R__v > 1 || (file && file->GetVersion() == 22304)) {
         fFitFormat.Streamer(R__b);
         fStatFormat.Streamer(R__b);
      } else {
         SetFitFormat();
         SetStatFormat();
      }
      R__b.CheckByteCount(R__s, R__c, TPaveStats::IsA());
      //====end of old versions

   } else {
      TPaveStats::Class()->WriteBuffer(R__b,this);
   }
}
