// @(#)root/graf:$Id$
// Author: Rene Brun   17/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TPaveLabel.h"
#include "TLatex.h"
#include "TVirtualPad.h"

ClassImp(TPaveLabel)


//______________________________________________________________________________
//*  A PaveLabel is a Pave (see TPave) with a text centered in the Pave.
//Begin_Html
/*
<img src="gif/pavelabel.gif">
*/
//End_Html
//


//______________________________________________________________________________
TPaveLabel::TPaveLabel(): TPave(), TAttText()
{
   // Pavelabel default constructor.
}


//______________________________________________________________________________
TPaveLabel::TPaveLabel(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, const char *label, Option_t *option)
           :TPave(x1,y1,x2,y2,3,option), TAttText(22,0,1,gStyle->GetTextFont(),0.99)
{
   // Pavelabel normal constructor.
   //
   // a PaveLabel is a Pave with a label centered in the Pave
   // The Pave is by default defined bith bordersize=5 and option ="br".
   // The text size is automatically computed as a function of the pave size.
   //
   //  IMPORTANT NOTE:
   //  Because TPave objects (and objects deriving from TPave) have their
   //  master coordinate system in NDC, one cannot use the TBox functions
   //  SetX1,SetY1,SetX2,SetY2 to change the corner coordinates. One should use
   //  instead SetX1NDC, SetY1NDC, SetX2NDC, SetY2NDC.

   fLabel  = label;
}


//______________________________________________________________________________
TPaveLabel::~TPaveLabel()
{
   // Pavelabel default destructor.
}


//______________________________________________________________________________
TPaveLabel::TPaveLabel(const TPaveLabel &pavelabel) : TPave(pavelabel), TAttText(pavelabel)
{
   // Pavelabel copy constructor.

   ((TPaveLabel&)pavelabel).Copy(*this);
}


//______________________________________________________________________________
void TPaveLabel::Copy(TObject &obj) const
{
   // Copy this pavelabel to pavelabel.

   TPave::Copy(obj);
   TAttText::Copy(((TPaveLabel&)obj));
   ((TPaveLabel&)obj).fLabel      = fLabel;
}


//______________________________________________________________________________
void TPaveLabel::Draw(Option_t *option)
{
   // Draw this pavelabel with its current attributes.

   Option_t *opt;
   if (option && strlen(option)) opt = option;
   else                          opt = GetOption();

   AppendPad(opt);
}


//______________________________________________________________________________
void TPaveLabel::DrawPaveLabel(Double_t x1, Double_t y1, Double_t x2, Double_t y2, const char *label, Option_t *option)
{
   // Draw this pavelabel with new coordinates.

   TPaveLabel *newpavelabel = new TPaveLabel(x1,y1,x2,y2,label,option);
   newpavelabel->SetBit(kCanDelete);
   newpavelabel->AppendPad();
}


//______________________________________________________________________________
void TPaveLabel::Paint(Option_t *option)
{
   // Paint this pavelabel with its current attributes.

   // Convert from NDC to pad coordinates
   TPave::ConvertNDCtoPad();

   PaintPaveLabel(fX1, fY1, fX2, fY2, GetLabel(), strlen(option)?option:GetOption());
}


//______________________________________________________________________________
void TPaveLabel::PaintPaveLabel(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                      const char *label ,Option_t *option)
{
   // Draw this pavelabel with new coordinates.

   Int_t nch = strlen(label);

   // Draw the pave
   TPave::PaintPave(x1,y1,x2,y2,GetBorderSize(),option);

   Float_t nspecials = 0;
   for (Int_t i=0;i<nch;i++) {
      if (label[i] == '!') nspecials += 1;
      if (label[i] == '?') nspecials += 1.5;
      if (label[i] == '#') nspecials += 1;
      if (label[i] == '`') nspecials += 1;
      if (label[i] == '^') nspecials += 1.5;
      if (label[i] == '~') nspecials += 1;
      if (label[i] == '&') nspecials += 2;
      if (label[i] == '\\') nspecials += 3;  // octal characters very likely
   }
   nch -= Int_t(nspecials + 0.5);
   if (nch <= 0) return;

   // Draw label
   Double_t wh   = (Double_t)gPad->XtoPixel(gPad->GetX2());
   Double_t hh   = (Double_t)gPad->YtoPixel(gPad->GetY1());
   Double_t labelsize, textsize = GetTextSize();
   Int_t automat = 0;
   if (GetTextFont()%10 > 2) {  // fixed size font specified in pixels
      labelsize = GetTextSize();
   } else {
      if (TMath::Abs(textsize -0.99) < 0.001) automat = 1;
      if (textsize == 0)   { textsize = 0.99; automat = 1;}
      Int_t ypixel      = TMath::Abs(gPad->YtoPixel(y1) - gPad->YtoPixel(y2));
      labelsize = textsize*ypixel/hh;
      if (wh < hh) labelsize *= hh/wh;
   }
   TLatex latex;
   latex.SetTextAngle(GetTextAngle());
   latex.SetTextFont(GetTextFont());
   latex.SetTextAlign(GetTextAlign());
   latex.SetTextColor(GetTextColor());
   latex.SetTextSize(labelsize);
   if (automat) {
      UInt_t w,h;
      latex.GetTextExtent(w,h,GetTitle());
      labelsize = h/hh;
      Double_t wxlabel   = TMath::Abs(gPad->XtoPixel(x2) - gPad->XtoPixel(x1));
      while (w > 0.99*wxlabel) { labelsize *= 0.99*wxlabel/w;latex.SetTextSize(labelsize); latex.GetTextExtent(w,h,GetTitle());}
      if (h < 1) h = 1;
      if (h==1) {
      labelsize   = Double_t(h)/hh;
      if (wh < hh) labelsize *= hh/wh;
      latex.SetTextSize(labelsize);
      }
   }
   Int_t halign = GetTextAlign()/10;
   Int_t valign = GetTextAlign()%10;
   Double_t x = 0.5*(x1+x2);
   if (halign == 1) x = x1 + 0.02*(x2-x1);
   if (halign == 3) x = x2 - 0.02*(x2-x1);
   Double_t y = 0.5*(y1+y2);
   if (valign == 1) y = y1 + 0.02*(y2-y1);
   if (valign == 3) y = y2 - 0.02*(y2-y1);
   latex.PaintLatex(x, y, GetTextAngle(),labelsize,GetLabel());
}


//______________________________________________________________________________
void TPaveLabel::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPaveLabel::Class())) {
      out<<"   ";
   } else {
      out<<"   TPaveLabel *";
   }
   TString s = fLabel.Data();
   s.ReplaceAll("\"","\\\"");
   if (fOption.Contains("NDC")) {
      out<<"pl = new TPaveLabel("<<fX1NDC<<","<<fY1NDC<<","<<fX2NDC<<","<<fY2NDC
         <<","<<quote<<s.Data()<<quote<<","<<quote<<fOption<<quote<<");"<<endl;
   } else {
      out<<"pl = new TPaveLabel("<<gPad->PadtoX(fX1)<<","<<gPad->PadtoY(fY1)<<","<<gPad->PadtoX(fX2)<<","<<gPad->PadtoY(fY2)
         <<","<<quote<<s.Data()<<quote<<","<<quote<<fOption<<quote<<");"<<endl;
   }
   if (fBorderSize != 3) {
      out<<"   pl->SetBorderSize("<<fBorderSize<<");"<<endl;
   }
   SaveFillAttributes(out,"pl",19,1001);
   SaveLineAttributes(out,"pl",1,1,1);
   SaveTextAttributes(out,"pl",22,0,1,62,0);

   out<<"   pl->Draw();"<<endl;
}
