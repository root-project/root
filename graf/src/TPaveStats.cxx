// @(#)root/graf:$Name:  $:$Id: TPaveStats.cxx,v 1.29 2007/04/15 06:08:27 brun Exp $
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
#include "TROOT.h"
#include "TPaveStats.h"
#include "TPaveLabel.h"
#include "TVirtualPad.h"
#include "TStyle.h"
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
//  WARNING: never call SetOptStat(000111); but SetOptStat(1111), 0001111 will
//          be taken as an octal number !!
//  NOTE that in case of 2-D histograms, when selecting just underflow (10000)
//        or overflow (100000), the stats box will show all combinations
//        of underflow/overflows and not just one single number!
//

const UInt_t kTakeStyle = BIT(17); //see TStyle::SetOptFit/Stat


//______________________________________________________________________________
TPaveStats::TPaveStats(): TPaveText()
{
   // TPaveStats default constructor
   
   fParent = 0;
}


//______________________________________________________________________________
TPaveStats::TPaveStats(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Option_t *option)
           :TPaveText(x1,y1,x2,y2,option)
{
   // TPaveStats normal constructor

   fParent = 0;
   fOptFit  = gStyle->GetOptFit();
   fOptStat = gStyle->GetOptStat();
   SetFitFormat(gStyle->GetFitFormat());
   SetStatFormat(gStyle->GetStatFormat());
}


//______________________________________________________________________________
TPaveStats::~TPaveStats()
{
   // TPaveStats default destructor

   if ( fParent && !fParent->TestBit(kInvalidObject)) fParent->RecursiveRemove(this);
}


//______________________________________________________________________________
Int_t TPaveStats::GetOptFit() const 
{
   // return the fit option

   if (TestBit(kTakeStyle)) return gStyle->GetOptFit();
   return fOptFit;
}


//______________________________________________________________________________
Int_t TPaveStats::GetOptStat() const 
{
   // return the stat option

   if (TestBit(kTakeStyle)) return gStyle->GetOptStat();
   return fOptStat;
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
void TPaveStats::SetOptFit(Int_t fit)
{
   // set the fit option

   fOptFit = fit;
   ResetBit(kTakeStyle);
}


//______________________________________________________________________________
void TPaveStats::SetOptStat(Int_t stat)
{
   // set the stat option

   fOptStat = stat;
   ResetBit(kTakeStyle);
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
   // Paint the pave stat.

   TPave::ConvertNDCtoPad();
   TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);

   if (!fLines) return;
   Double_t dx = fX2 - fX1;
   Double_t dy = fY2 - fY1;
   Double_t titlesize=0;
   Double_t textsize = GetTextSize();
   Int_t nlines = GetSize();
   if (nlines == 0) nlines = 5;
   Int_t print_name = fOptStat%10;

   // Evaluate text size as a function of the number of lines
   Double_t y1       = gPad->GetY1();
   Double_t y2       = gPad->GetY2();
   Float_t margin    = fMargin*(fX2-fX1);
   Double_t yspace   = (fY2 - fY1)/Double_t(nlines);
   Double_t textsave = textsize;
   TObject *line;
   TLatex *latex, *latex_tok;
   TIter next(fLines);
   Double_t longest = 0, titlelength = 0;
   Double_t w, wtok[2];
   char *st, *sl=0;
   if (textsize == 0)  {
      //textsize = 0.85*yspace/(y2 - y1);
      textsize = 0.92*yspace/(y2 - y1);
      titlesize = textsize;
      wtok[0] = 0; wtok[1] = 0;
      while ((line = (TObject*) next())) {
         if (line->IsA() == TLatex::Class()) {
            latex = (TLatex*)line;
            sl = new char[strlen(latex->GetTitle())+1];
            strcpy(sl, latex->GetTitle());
            if (strpbrk(sl, "=") !=0 && print_name == 0) {
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
            } else if (strpbrk(sl, "|") !=0) {
            } else {
               print_name = 0;
               latex->SetTextSize(titlesize);
               titlelength = latex->GetXsize()+2.*margin;
               if (titlelength > 0.98*dx) titlesize *= 0.98*dx/titlelength;
            }
            delete [] sl; sl = 0;
         }
      }
      longest = wtok[0]+wtok[1]+2.*margin;
      if (longest > 0.98*dx) textsize *= 0.98*dx/longest;
      SetTextSize(textsize);
   } else {
      titlesize = textsize;
   }
   Double_t ytext = fY2 + 0.5*yspace;
   Double_t xtext = 0;
   print_name = fOptStat%10;

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
         // Draw all the histogram stats except the 2D under/overflow
         if (strpbrk(sl, "=") !=0 && print_name == 0) {
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
            Double_t yline1 = ytext+yspace/2.;
            Double_t yline2 = ytext-yspace/2.;
            Double_t xline1 = (fX2-fX1)/3+fX1;
            Double_t xline2 = 2*(fX2-fX1)/3+fX1;
            gPad->PaintLine(fX1,yline1,fX2,yline1);
            gPad->PaintLine(xline1,yline1,xline1,yline2);
            gPad->PaintLine(xline2,yline1,xline2,yline2);
            st = strtok(sl, "|");
            Int_t theIndex = 0;
            while ( st !=0 ) {
               latex->SetTextAlign(22);
               if (theIndex == 0) xtext = 0.5*(fX1+xline1);
               if (theIndex == 1) xtext = 0.5*(fX1+fX2);
               if (theIndex == 2) xtext = 0.5*(xline2+fX2);
               latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                             latex->GetTextSize(),
                                             st);
               theIndex++;
               st = strtok(0, "|");
            }
         // Draw the histogram identifier
         } else {
            print_name = 0;
            latex->SetTextAlign(22);
            xtext = 0.5*(fX1+fX2);
            latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                          titlesize,
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

   // if a label create & paint a pavetext title
   if (fLabel.Length() > 0) {
      Double_t x1,x2;
      dy = gPad->GetY2() - gPad->GetY1();
      x1 = fX1 + 0.25*dx;
      x2 = fX2 - 0.25*dx;
      y1 = fY2 - 0.02*dy;
      y2 = fY2 + 0.02*dy;
      TPaveLabel *title = new TPaveLabel(x1,y1,x2,y2,fLabel.Data(),GetDrawOption());
      title->SetFillColor(GetFillColor());
      title->SetTextColor(GetTextColor());
      title->SetTextFont(GetTextFont());
      title->Paint();
      delete title;
   }
}


//______________________________________________________________________________
void TPaveStats::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPaveStats::Class())) {
      out<<"   ";
   } else {
      out<<"   "<<ClassName()<<" *";
   }
   if (fOption.Contains("NDC")) {
      out<<"ptstats = new "<<ClassName()<<"("<<fX1NDC<<","<<fY1NDC<<","<<fX2NDC<<","<<fY2NDC
      <<","<<quote<<fOption<<quote<<");"<<endl;
   } else {
      out<<"ptstats = new "<<ClassName()<<"("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<quote<<fOption<<quote<<");"<<endl;
   }
   if (strcmp(GetName(),"TPave")) {
      out<<"   ptstats->SetName("<<quote<<GetName()<<quote<<");"<<endl;
   }
   if (fBorderSize != 4) {
      out<<"   ptstats->SetBorderSize("<<fBorderSize<<");"<<endl;
   }
   SaveFillAttributes(out,"ptstats",0,1001);
   SaveLineAttributes(out,"ptstats",1,1,1);
   SaveTextAttributes(out,"ptstats",22,0,1,62,0);
   SaveLines(out,"ptstats");
   out<<"   ptstats->SetOptStat("<<GetOptStat()<<");"<<endl;
   out<<"   ptstats->SetOptFit("<<GetOptFit()<<");"<<endl;
   out<<"   ptstats->Draw();"<<endl;
}


//______________________________________________________________________________
void TPaveStats::Streamer(TBuffer &R__b)
{
   // Stream an object of class TPaveStats.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         R__b.ReadClassBuffer(TPaveStats::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TPaveText::Streamer(R__b);
      R__b >> fOptFit;
      R__b >> fOptStat;
      if (R__v > 1 || R__b.GetVersionOwner() == 22304) {
         fFitFormat.Streamer(R__b);
         fStatFormat.Streamer(R__b);
      } else {
         SetFitFormat();
         SetStatFormat();
      }
      R__b.CheckByteCount(R__s, R__c, TPaveStats::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TPaveStats::Class(),this);
   }
}


//______________________________________________________________________________
void TPaveStats::UseCurrentStyle()
{
   // Replace current attributes by current style.

   if (gStyle->IsReading()) {
      SetOptStat(gStyle->GetOptStat());
      SetOptFit(gStyle->GetOptFit());
      SetStatFormat(gStyle->GetStatFormat());
      SetFitFormat(gStyle->GetFitFormat());
      SetBorderSize(gStyle->GetStatBorderSize());
      SetFillColor(gStyle->GetStatColor());
      SetFillStyle(gStyle->GetStatStyle());
      SetTextFont(gStyle->GetStatFont());
      SetTextSize(gStyle->GetStatFontSize());
      SetTextColor(gStyle->GetStatTextColor());
      SetX2NDC(gStyle->GetStatX());
      SetY2NDC(gStyle->GetStatY());
      SetX1NDC(gStyle->GetStatX()-gStyle->GetStatW());
      SetY1NDC(gStyle->GetStatY()-gStyle->GetStatH());
   } else {
      gStyle->SetOptStat(GetOptStat());
      gStyle->SetOptFit(GetOptFit());
      gStyle->SetStatFormat(GetStatFormat());
      gStyle->SetFitFormat(GetFitFormat());
      gStyle->SetStatBorderSize(GetBorderSize());
      gStyle->SetTextColor(GetTextColor());
      gStyle->SetStatColor(GetFillColor());
      gStyle->SetStatStyle(GetFillStyle());
      gStyle->SetTextFont(GetTextFont());
      gStyle->SetStatFontSize(GetTextSize());
      gStyle->SetStatTextColor(GetTextColor());
      gStyle->SetStatX(GetX2NDC());
      gStyle->SetStatY(GetY2NDC());
      gStyle->SetStatW(GetX2NDC()-GetX1NDC());
      gStyle->SetStatH(GetY2NDC()-GetY1NDC());
   }
}
