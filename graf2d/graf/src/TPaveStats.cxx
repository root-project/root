// @(#)root/graf:$Id$
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
/* Begin_Html
<center><h2>The histogram statistics painter class</h2></center>
A PaveStats is a PaveText to draw histogram statistics and fit parameters.
<ul>
<li><a href="#PS01">Statistics Display</a>
<li><a href="#PS02">Fit Statistics</a>
<li><a href="#PS03">Statistics box editing</a>
</ul>

<a name="PS01"></a><h3>Statistics Display</h3>

The type of information shown in the histogram statistics box can be selected
with:
<pre>
      gStyle->SetOptStat(mode);
</pre>

<p>The "<tt>mode</tt>" has up to nine digits that can be set to on (1 or 2), off (0).
<pre>
      mode = ksiourmen  (default = 000001111)
      k = 1;  kurtosis printed
      k = 2;  kurtosis and kurtosis error printed
      s = 1;  skewness printed
      s = 2;  skewness and skewness error printed
      i = 1;  integral of bins printed
      o = 1;  number of overflows printed
      u = 1;  number of underflows printed
      r = 1;  rms printed
      r = 2;  rms and rms error printed
      m = 1;  mean value printed
      m = 2;  mean and mean error values printed
      e = 1;  number of entries printed
      n = 1;  name of histogram is printed
</pre>

<p>For example:
<pre>
      gStyle->SetOptStat(11);
</pre>
displays only the name of histogram and the number of entries, whereas:
<pre>
      gStyle->SetOptStat(1101);
</pre>
displays the name of histogram, mean value and RMS.

<p><b>WARNING 1:</b> never do:
<pre>
      <s>gStyle->SetOptStat(0001111);</s>
</pre>
but instead do:
<pre>
      gStyle->SetOptStat(1111);
</pre>
because <tt>0001111</tt> will be taken as an octal number!

<p><b>WARNING 2:</b> for backward compatibility with older versions
<pre>
      gStyle->SetOptStat(1);
</pre>
is taken as:
<pre>
      gStyle->SetOptStat(1111)
</pre>
To print only the name of the histogram do:
<pre>
      gStyle->SetOptStat(1000000001);
</pre>

<p><b>NOTE</b> that in case of 2D histograms, when selecting only underflow
(10000) or overflow (100000), the statistics box will show all combinations
of underflow/overflows and not just one single number.

<p>The parameter mode can be any combination of the letters <tt>kKsSiourRmMen</tt>
<pre>
      k :  kurtosis printed
      K :  kurtosis and kurtosis error printed
      s :  skewness printed
      S :  skewness and skewness error printed
      i :  integral of bins printed
      o :  number of overflows printed
      u :  number of underflows printed
      r :  rms printed
      R :  rms and rms error printed
      m :  mean value printed
      M :  mean value mean error values printed
      e :  number of entries printed
      n :  name of histogram is printed
</pre>

<p>For example, to print only name of histogram and number of entries do:
<pre>
      gStyle->SetOptStat("ne");
</pre>

<p>To print only the name of the histogram do:
<pre>
      gStyle->SetOptStat("n");
</pre>

<p>The default value is:
<pre>
      gStyle->SetOptStat("nemr");
</pre>

<p>When a histogram is painted, a <tt>TPaveStats</tt> object is created and added
to the list of functions of the histogram. If a <tt>TPaveStats</tt> object
already exists in the histogram list of functions, the existing object is just
updated with the current histogram parameters.

<p>Once a histogram is painted, the statistics box can be accessed using
<tt>h->FindObject("stats")</tt>. In the command line it is enough to do:
<pre>
      Root > h->Draw()
      Root > TPaveStats *st = (TPaveStats*)h->FindObject("stats")
</pre>

<p>because after <tt>h->Draw()</tt> the histogram is automatically painted. But
in a script file the painting should be forced using <tt>gPad->Update()</tt>
in order to make sure the statistics box is created:
<pre>
      h->Draw();
      gPad->Update();
      TPaveStats *st = (TPaveStats*)h->FindObject("stats");
</pre>

<p>Without <tt>gPad->Update()</tt> the line <tt>h->FindObject("stats")</tt>
returns a null pointer.

<p>When a histogram is drawn with the option "<tt>SAME</tt>", the statistics box
is not drawn. To force the statistics box drawing with the option
"<tt>SAME</tt>", the option "<tt>SAMES</tt>" must be used.
If the new statistics box hides the previous statistics box, one can change
its position with these lines ("<tt>h</tt>" being the pointer to the histogram):
<pre>
      Root > TPaveStats *st = (TPaveStats*)h->FindObject("stats")
      Root > st->SetX1NDC(newx1); //new x start position
      Root > st->SetX2NDC(newx2); //new x end position
</pre>

<p>To change the type of information for an histogram with an existing
<tt>TPaveStats</tt> one should do:
<pre>
      st->SetOptStat(mode);
</pre>
Where "<tt>mode</tt>" has the same meaning than when calling
<tt>gStyle->SetOptStat(mode)</tt> (see above).

<p>One can delete the statistics box for a histogram <tt>TH1* h</tt> with:
<pre>
      h->SetStats(0)
</pre>

<p>and activate it again with:
<pre>
      h->SetStats(1).
</pre>


<a name="PS02"></a><h3>Fit Statistics</h3>


The type of information about fit parameters printed in the histogram statistics
box can be selected via the parameter mode. The parameter mode can be
<tt>= pcev</tt>  (default <tt>= 0111</tt>)
<pre>
      p = 1;  print Probability
      c = 1;  print Chisquare/Number of degrees of freedom
      e = 1;  print errors (if e=1, v must be 1)
      v = 1;  print name/values of parameters
</pre>
Example:
<pre>
      gStyle->SetOptFit(1011);
</pre>
print fit probability, parameter names/values and errors.
<ol>
<li> When <tt>"v" = 1</tt> is specified, only the non-fixed parameters are
     shown.
<li> When <tt>"v" = 2</tt> all parameters are shown.
</ol>
Note: <tt>gStyle->SetOptFit(1)</tt> means "default value", so it is equivalent
to <tt>gStyle->SetOptFit(111)</tt>


<a name="PS03"></a><h3>Statistics box editing</h3>

The following example show how to remove and add a line in a statistics box.
End_Html
Begin_Macro(source)
../../../tutorials/hist/statsEditing.C
End_Macro */


const UInt_t kTakeStyle = BIT(17); //see TStyle::SetOptFit/Stat


//______________________________________________________________________________
TPaveStats::TPaveStats(): TPaveText()
{
   /* Begin_Html
   TPaveStats default constructor.
   End_Html */

   fParent  = 0;
   fOptFit  = gStyle->GetOptFit();
   fOptStat = gStyle->GetOptStat();
}


//______________________________________________________________________________
TPaveStats::TPaveStats(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Option_t *option)
           :TPaveText(x1,y1,x2,y2,option)
{
   /* Begin_Html
   TPaveStats normal constructor.
   End_Html */

   fParent = 0;
   fOptFit  = gStyle->GetOptFit();
   fOptStat = gStyle->GetOptStat();
   SetFitFormat(gStyle->GetFitFormat());
   SetStatFormat(gStyle->GetStatFormat());
}


//______________________________________________________________________________
TPaveStats::~TPaveStats()
{
   /* Begin_Html
   TPaveStats default destructor.
   End_Html */

   if ( fParent && !fParent->TestBit(kInvalidObject)) fParent->RecursiveRemove(this);
}


//______________________________________________________________________________
Int_t TPaveStats::GetOptFit() const
{
   /* Begin_Html
   Return the fit option.
   End_Html */

   if (TestBit(kTakeStyle)) return gStyle->GetOptFit();
   return fOptFit;
}


//______________________________________________________________________________
Int_t TPaveStats::GetOptStat() const
{
   /* Begin_Html
   Return the stat option.
   End_Html */

   if (TestBit(kTakeStyle)) return gStyle->GetOptStat();
   return fOptStat;
}


//______________________________________________________________________________
void TPaveStats::SaveStyle()
{
   /* Begin_Html
   Save This TPaveStats options in current style.
   End_Html */

   gStyle->SetOptFit(fOptFit);
   gStyle->SetOptStat(fOptStat);
   gStyle->SetFitFormat(fFitFormat.Data());
   gStyle->SetStatFormat(fStatFormat.Data());
}


//______________________________________________________________________________
void TPaveStats::SetFitFormat(const char *form)
{
   /* Begin_Html
   Change (i.e. set) the format for printing fit parameters in statistics box.
   End_Html */

   fFitFormat = form;
}


//______________________________________________________________________________
void TPaveStats::SetOptFit(Int_t fit)
{
   /* Begin_Html
   Set the fit option.
   End_Html */

   fOptFit = fit;
   ResetBit(kTakeStyle);
}


//______________________________________________________________________________
void TPaveStats::SetOptStat(Int_t stat)
{
   /* Begin_Html
   Set the stat option.
   End_Html */

   fOptStat = stat;
   ResetBit(kTakeStyle);
}


//______________________________________________________________________________
void TPaveStats::SetStatFormat(const char *form)
{
   /* Begin_Html
   Change (i.e. set) the format for printing statistics.
   End_Html */

   fStatFormat = form;
}


//______________________________________________________________________________
void TPaveStats::Paint(Option_t *option)
{
   /* Begin_Html
   Paint the pave stat.
   End_Html */

   TPave::ConvertNDCtoPad();
   TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);

   if (!fLines) return;
   TString typolabel;
   Double_t y2ref = TMath::Max(fY1,fY2);
   Double_t x1ref = TMath::Min(fX1,fX2);
   Double_t x2ref = TMath::Max(fX1,fX2);
   Double_t dx    = TMath::Abs(fX2 - fX1);
   Double_t dy    = TMath::Abs(fY2 - fY1);
   Double_t titlesize=0;
   Double_t textsize = GetTextSize();
   Int_t nlines = GetSize();
   if (nlines == 0) nlines = 5;
   Int_t print_name = fOptStat%10;

   // Evaluate text size as a function of the number of lines
   Double_t y1       = gPad->GetY1();
   Double_t y2       = gPad->GetY2();
   Float_t margin    = fMargin*dx;
   Double_t yspace   = dy/Double_t(nlines);
   Double_t textsave = textsize;
   TObject *line;
   TLatex *latex, *latex_tok;
   TIter next(fLines);
   Double_t longest = 0, titlelength = 0;
   Double_t w, wtok[2];
   char *st, *sl=0;
   if (textsize == 0)  {
      textsize = 0.92*yspace/(y2 - y1);
      titlesize = textsize;
      wtok[0] = 0; wtok[1] = 0;
      while ((line = (TObject*) next())) {
         if (line->IsA() == TLatex::Class()) {
            latex = (TLatex*)line;
            Int_t nchs = strlen(latex->GetTitle());
            sl = new char[nchs+1];
            strlcpy(sl, latex->GetTitle(),nchs+1);
            if (strpbrk(sl, "=") !=0 && print_name == 0) {
               st = strtok(sl, "=");
               Int_t itok = 0;
               while ( st !=0 ) {
                  latex_tok = new TLatex(0.,0.,st);
                  Style_t tfont = latex->GetTextFont();
                  if (tfont == 0) tfont = GetTextFont();
                  latex_tok->SetTextFont(tfont);
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
               Style_t tfont = latex->GetTextFont();
               if (tfont == 0) latex->SetTextFont(GetTextFont());
               latex->SetTextSize(titlesize);
               titlelength = latex->GetXsize()+2.*margin;
               if (titlelength > 0.98*dx) titlesize *= 0.98*dx/titlelength;
               latex->SetTextFont(tfont);
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
   Double_t ytext = y2ref + 0.5*yspace;
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

         Int_t nchs = strlen(latex->GetTitle());
         sl = new char[nchs+1];
         strlcpy(sl, latex->GetTitle(),nchs+1);
         // Draw all the histogram stats except the 2D under/overflow
         if (strpbrk(sl, "=") !=0 && print_name == 0) {
            st = strtok(sl, "=");
            Int_t halign = 12;
            while ( st !=0 ) {
               typolabel = st;
               latex->SetTextAlign(halign);
               if (halign == 12) xtext = x1ref + margin;
               if (halign == 32) {
                  xtext = x2ref - margin;
                  typolabel = typolabel.Strip();
                  typolabel.ReplaceAll("-","#minus");
               }
               latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                             latex->GetTextSize(),
                                             typolabel.Data());
               st = strtok(0, "=");
               halign = 32;
            }
         // Draw the 2D under/overflow
         } else if (strpbrk(sl, "|") !=0) {
            Double_t yline1 = ytext+yspace/2.;
            Double_t yline2 = ytext-yspace/2.;
            Double_t xline1 = dx/3+x1ref;
            Double_t xline2 = 2*dx/3+x1ref;
            gPad->PaintLine(x1ref,yline1,x2ref,yline1);
            gPad->PaintLine(xline1,yline1,xline1,yline2);
            gPad->PaintLine(xline2,yline1,xline2,yline2);
            st = strtok(sl, "|");
            Int_t theIndex = 0;
            while ( st !=0 ) {
               latex->SetTextAlign(22);
               if (theIndex == 0) xtext = 0.5*(x1ref+xline1);
               if (theIndex == 1) xtext = 0.5*(x1ref+x2ref);
               if (theIndex == 2) xtext = 0.5*(xline2+x2ref);
               typolabel = st;
               typolabel.ReplaceAll("-", "#minus");
               latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                             latex->GetTextSize(),
                                             typolabel.Data());
               theIndex++;
               st = strtok(0, "|");
            }
         // Draw the histogram identifier
         } else {
            print_name = 0;
            latex->SetTextAlign(22);
            xtext = 0.5*(x1ref+x2ref);
            latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                                          titlesize,
                                          sl);
            gPad->PaintLine(x1ref,y2ref-yspace,x2ref,y2ref-yspace);
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
      x1 = x1ref + 0.25*dx;
      x2 = x2ref - 0.25*dx;
      y1 = y2ref - 0.02*dy;
      y2 = y2ref + 0.02*dy;
      TPaveLabel *title = new TPaveLabel(x1,y1,x2,y2,fLabel.Data(),GetDrawOption());
      title->SetFillColor(GetFillColor());
      title->SetTextColor(GetTextColor());
      title->SetTextFont(GetTextFont());
      title->Paint();
      delete title;
   }
}


//______________________________________________________________________________
void TPaveStats::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   /* Begin_Html
   Save primitive as a C++ statement(s) on output stream out.
   End_Html */

   char quote = '"';
   out<<"   "<<std::endl;
   if (gROOT->ClassSaved(TPaveStats::Class())) {
      out<<"   ";
   } else {
      out<<"   "<<ClassName()<<" *";
   }
   if (fOption.Contains("NDC")) {
      out<<"ptstats = new "<<ClassName()<<"("<<fX1NDC<<","<<fY1NDC<<","<<fX2NDC<<","<<fY2NDC
      <<","<<quote<<fOption<<quote<<");"<<std::endl;
   } else {
      out<<"ptstats = new "<<ClassName()<<"("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<quote<<fOption<<quote<<");"<<std::endl;
   }
   if (strcmp(GetName(),"TPave")) {
      out<<"   ptstats->SetName("<<quote<<GetName()<<quote<<");"<<std::endl;
   }
   if (fBorderSize != 4) {
      out<<"   ptstats->SetBorderSize("<<fBorderSize<<");"<<std::endl;
   }
   SaveFillAttributes(out,"ptstats",19,1001);
   SaveLineAttributes(out,"ptstats",1,1,1);
   SaveTextAttributes(out,"ptstats",22,0,1,62,0);
   SaveLines(out,"ptstats");
   out<<"   ptstats->SetOptStat("<<GetOptStat()<<");"<<std::endl;
   out<<"   ptstats->SetOptFit("<<GetOptFit()<<");"<<std::endl;
   out<<"   ptstats->Draw();"<<std::endl;
}


//______________________________________________________________________________
void TPaveStats::Streamer(TBuffer &R__b)
{
   /* Begin_Html
   Stream an object of class TPaveStats.
   End_Html */

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
   /* Begin_Html
   Replace current attributes by current style.
   End_Html */

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
