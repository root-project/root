// @(#)root/graf:$Id$
// Author: Rene Brun   20/10/95

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
#include "TBufferFile.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TPaveLabel.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TLatex.h"
#include "TError.h"
#include "TColor.h"
#include "TClass.h"

ClassImp(TPaveText)


//______________________________________________________________________________
/* Begin_Html
<center><h2>TPaveText : to draw a pave with text</h2></center>
A PaveText is a Pave (see TPave) with text, lines or/and boxes inside.
Line (and boxes) are positionned in the pave using coordinates relative to
the pave (%).
<p>
Example:
End_Html
Begin_Macro(source)
../../../tutorials/graphics/pavetext.C
End_Macro */



//______________________________________________________________________________
TPaveText::TPaveText(): TPave(), TAttText()
{
   // pavetext default constructor.

   fLines   = 0;
   fMargin  = 0.05;
   fLongest = 0;
}


//______________________________________________________________________________
TPaveText::TPaveText(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Option_t *option)
           :TPave(x1,y1,x2,y2,4,option), TAttText(22,0,gStyle->GetTextColor(),gStyle->GetTextFont(),0)
{
   // PaveText normal constructor.
   //
   // A PaveText is a Pave with several lines of text
   //
   //  option = "TR" Top and Right shadows are drawn.
   //  option = "TL" Top and Left shadows are drawn.
   //  option = "BR" Bottom and Right shadows are drawn.
   //  option = "BL" Bottom and Left shadows are drawn.
   //
   //  If none of these four above options is specified the default the
   //  option "BR" will be used to draw the border. To produces a pave
   //  without any border it is enough to specify the option "NB" (no border).
   //
   //  option = "NDC" x1,y1,x2,y2 are given in NDC
   //  option = "ARC" corners are rounded
   //
   // In case of option "ARC", the corner radius is specified
   // via TPave::SetCornerRadius(rad) where rad is given in percent
   // of the pave height (default value is 0.2).
   //
   // The individual text items are entered via AddText
   // By default, text items inherits from the default pavetext AttText.
   // A title can be added later to this pavetext via TPaveText::SetLabel.

   fLines   = new TList;
   fMargin  = 0.05;
   fLongest = 0;
}


//______________________________________________________________________________
TPaveText::~TPaveText()
{
   // pavetext default destructor.

   if (!TestBit(kNotDeleted)) return;
   if (fLines) fLines->Delete();
   delete fLines;
   fLines = 0;
}


//______________________________________________________________________________
TPaveText::TPaveText(const TPaveText &pavetext) : TPave(), TAttText()
{
   // pavetext copy constructor.

   TBufferFile b(TBuffer::kWrite);
   TPaveText *p = (TPaveText*)(&pavetext);
   p->Streamer(b);
   b.SetReadMode();
   b.SetBufferOffset(0);
   fLines = 0;
   Streamer(b);
}

//______________________________________________________________________________
TPaveText& TPaveText::operator=(const TPaveText& pt)
{
   //assignment operator
   if(this!=&pt) {
      TPave::operator=(pt);
      TAttText::operator=(pt);
      fLabel=pt.fLabel;
      fLongest=pt.fLongest;
      fMargin=pt.fMargin;
      fLines=pt.fLines;
   }
   return *this;
}

//______________________________________________________________________________
TBox *TPaveText::AddBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Add a new graphics box to this pavetext.

   if (!gPad->IsEditable()) return 0;
   TBox *newbox = new TBox(x1,y1,x2,y2);

   if (!fLines) fLines = new TList;
   fLines->Add(newbox);
   return newbox;
}


//______________________________________________________________________________
TLine *TPaveText::AddLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   // Add a new graphics line to this pavetext.

   if (!gPad->IsEditable()) return 0;
   TLine *newline = new TLine(x1,y1,x2,y2);

   if (!fLines) fLines = new TList;
   fLines->Add(newline);
   return newline;
}


//______________________________________________________________________________
TText *TPaveText::AddText(Double_t x1, Double_t y1, const char *text)
{
   // Add a new Text line to this pavetext at given coordinates.

   TLatex *newtext = new TLatex(x1,y1,text);
   newtext->SetTextAlign(0);
   newtext->SetTextColor(0);
   newtext->SetTextFont(0);
   newtext->SetTextSize(0);
   Int_t nch = strlen(text);
   if (nch > fLongest) fLongest = nch;

   if (!fLines) fLines = new TList;
   fLines->Add(newtext);
   return newtext;
}


//______________________________________________________________________________
TText *TPaveText::AddText(const char *text)
{
   // Add a new Text line to this pavetext.

   return AddText(0,0,text);
}


//______________________________________________________________________________
void TPaveText::Clear(Option_t *)
{
   // Clear all lines in this pavetext.

   if (!fLines) return;
   fLines->Delete();
   fLongest = 0;
}


//______________________________________________________________________________
void TPaveText::DeleteText()
{
   // Delete text at the mouse position.

   if (!gPad->IsEditable()) return;
   if (!fLines) return;
   Double_t ymouse, yobj;
   TObject *obj = GetObject(ymouse, yobj);             //get object pointed by the mouse
   if (!obj) return;
   if (!obj->InheritsFrom(TText::Class())) return;
   fLines->Remove(obj);
   delete obj;
}


//______________________________________________________________________________
void TPaveText::Draw(Option_t *option)
{
   // Draw this pavetext with its current attributes.

   Option_t *opt;
   if (option && strlen(option)) opt = option;
   else                          opt = GetOption();

   AppendPad(opt);
}


//______________________________________________________________________________
void TPaveText::DrawFile(const char *filename, Option_t *option)
{
   // Draw lines in filename in this pavetext.

   ReadFile(filename);

   AppendPad(option);
}


//______________________________________________________________________________
void TPaveText::EditText()
{
   // Edit text at the mouse position.

   if (!gPad->IsEditable()) return;
   Double_t ymouse, yobj;
   TObject *obj = GetObject(ymouse, yobj);             //get object pointed by the mouse
   if (!obj) return;
   if (!obj->InheritsFrom(TText::Class())) return;
   TText *text = (TText*)obj;
   gROOT->SetSelectedPrimitive(text);
   gROOT->ProcessLine(Form("((TCanvas*)0x%lx)->SetSelected((TObject*)0x%lx)",
                           (ULong_t)gPad->GetCanvas(), (ULong_t)text));
   gROOT->ProcessLine(Form("((TCanvas*)0x%lx)->Selected((TVirtualPad*)0x%lx,(TObject*)0x%lx,1)",
                           (ULong_t)gPad->GetCanvas(), (ULong_t)gPad, (ULong_t)text));
   text->SetTextAttributes();
}


//______________________________________________________________________________
TText *TPaveText::GetLine(Int_t number) const
{
   // Get Pointer to line number in this pavetext.

   TText *line;
   TIter next(fLines);
   Int_t nlines = 0;
   while ((line = (TText*) next())) {
      if (nlines == number) return line;
      nlines++;
   }
   return 0;
}


//______________________________________________________________________________
TText *TPaveText::GetLineWith(const char *text) const
{
   // Get Pointer to first containing string text in this pavetext.

   TText *line;
   TIter next(fLines);
   while ((line = (TText*) next())) {
      if (strstr(line->GetTitle(),text)) return line;
   }
   return 0;
}


//______________________________________________________________________________
TObject *TPaveText::GetObject(Double_t &ymouse, Double_t &yobj) const
{
   // Get object pointed by the mouse in this pavetext.

   if (!fLines) return 0;
   Int_t nlines = GetSize();
   if (nlines == 0) return 0;

   // Evaluate text size as a function of the number of lines

   ymouse   = gPad->AbsPixeltoY(gPad->GetEventY());
   Double_t yspace   = (fY2 - fY1)/Double_t(nlines);
   Double_t textsize = GetTextSize();
   Double_t y1,y,dy;
   if (textsize == 0)  {
      y1       = gPad->GetY1();
   }
   Double_t ytext = fY2 + 0.5*yspace;
   Int_t valign;

   // Iterate over all lines
   // Copy pavetext attributes to line attributes if line attributes not set
   dy = fY2 - fY1;
   TObject *line;
   TText *linet;
   TLine *linel;
   TBox  *lineb;
   TIter next(fLines);
   while ((line = (TObject*) next())) {
   // Next primitive is a line
      if (line->IsA() == TLine::Class()) {
         linel = (TLine*)line;
         y1 = linel->GetY1();   if (y1 == 0) y1 = ytext; else y1 = fY1 + y1*dy;
         if (TMath::Abs(y1-ymouse) < 0.2*yspace) {yobj = y1; return line;}
         continue;
      }
   // Next primitive is a box
      if (line->IsA() == TBox::Class()) {
         lineb = (TBox*)line;
         y1 = lineb->GetY1();   if (y1 == 0) y1 = ytext; else y1 = fY1 + y1*dy;
         if (TMath::Abs(y1-ymouse) < 0.4*yspace) {yobj = y1; return line;}
         continue;
      }
   // Next primitive is a text
      if (line->InheritsFrom(TText::Class())) {
         linet = (TText*)line;
         ytext -= yspace;
         Double_t yl     = linet->GetY();
         Short_t talign = linet->GetTextAlign();
         if (talign == 0) talign = GetTextAlign();
         if (yl > 0 && yl <1) {
            ytext = fY1 + yl*dy;
         }
         valign = linet->GetTextAlign()%10;
         y = ytext;
         if (valign == 1) y = ytext -0.5*yspace;
         if (valign == 3) y = ytext +0.5*yspace;

         if (TMath::Abs(y-ymouse) < 0.5*yspace) {yobj = y; return line;}
      }
   }
   return 0;
}


//______________________________________________________________________________
Int_t TPaveText::GetSize() const
{
   //  return number of text lines (ignoring Tlines, etc)

   Int_t nlines = 0;
   TIter next(fLines);
   TObject *line;
   while ((line = (TObject*) next())) {
      if (line->InheritsFrom(TText::Class())) nlines++;
   }
   return nlines;
}


//______________________________________________________________________________
void TPaveText::InsertLine()
{
   // Add a new lineine at the mouse position.

   if (!gPad->IsEditable()) return;
   Double_t ymouse=0, yobj;
   TObject *obj = GetObject(ymouse, yobj); //get object pointed by the mouse
   Double_t yline = (ymouse-fY1)/(fY2-fY1);
   TLine *newline = AddLine(0,yline,0,yline);
   if (obj) {
      fLines->Remove(newline);        //remove line from last position
      if (yobj < ymouse) fLines->AddBefore(obj,newline);
      else               fLines->AddAfter(obj,newline);
   }
}


//______________________________________________________________________________
void TPaveText::InsertText(const char *text)
{
   // Add a new Text line at the mouse position.

   if (!gPad->IsEditable()) return;
   Double_t ymouse, yobj;
   TObject *obj = GetObject(ymouse, yobj); //get object pointed by the mouse
   TText *newtext = AddText(0,0,text);     //create new text object
   if (obj) {
      fLines->Remove(newtext);        //remove text from last position
      if (yobj < ymouse) fLines->AddBefore(obj,newtext); //insert new text at right position
      else               fLines->AddAfter(obj,newtext);  //insert new text at right position
   }
}


//______________________________________________________________________________
void TPaveText::Paint(Option_t *option)
{
   // Paint this pavetext with its current attributes.

   // Draw the pave
   TPave::ConvertNDCtoPad();
   TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);
   PaintPrimitives(kPaveText);
}


//______________________________________________________________________________
void TPaveText::PaintPrimitives(Int_t mode)
{
   // Paint list of primitives in this pavetext.

   if (!fLines) return;
   Double_t dx = fX2 - fX1;
   Double_t dy = fY2 - fY1;
   Double_t textsize = GetTextSize();
   Int_t nlines = GetSize();
   if (nlines == 0) nlines = 5;

   // Evaluate text size as a function of the number of lines

   Double_t x1,y1,x2,y2;
   y1       = gPad->GetY1();
   y2       = gPad->GetY2();
   Float_t margin  = fMargin*dx;
   Double_t yspace = dy/Double_t(nlines);
   Double_t textsave = textsize;
   TObject *line;
   TText *linet;
   TLatex *latex;
   TIter next(fLines);
   Double_t longest = 0;
   Double_t w;
   if (textsize == 0)  {
      textsize = 0.85*yspace/(y2 - y1);
      while ((line = (TObject*) next())) {
         if (line->IsA() == TLatex::Class()) {
            latex = (TLatex*)line;
            Float_t tangle = latex->GetTextAngle();
            if (latex->GetTextSize() != 0) continue;
            Style_t tfont = latex->GetTextFont();
            if (tfont == 0) latex->SetTextFont(GetTextFont());
            latex->SetTextSize(textsize);
            w = latex->GetXsize();
            latex->SetTextSize(0);
            latex->SetTextAngle(tangle); //text angle was redefined in GetXsize !
            if (w > longest) longest = w;
            latex->SetTextFont(tfont);
         }
      }
      if (longest > 0.92*dx) textsize *= 0.92*dx/longest;
      if (mode == kDiamond) textsize *= 0.66;
      SetTextSize(textsize);
   }
   Double_t ytext = fY2 + 0.5*yspace;
   Double_t xtext = 0;
   Int_t halign;

   // Iterate over all lines
   // Copy pavetext attributes to line attributes if line attributes not set
   TLine *linel;
   TBox  *lineb;
   next.Reset();
   while ((line = (TObject*) next())) {
   // Next primitive is a line
      if (line->IsA() == TLine::Class()) {
         linel = (TLine*)line;
         x1 = linel->GetX1();   if (x1 == 0) x1 = fX1; else x1 = fX1 + x1*dx;
         x2 = linel->GetX2();   if (x2 == 0) x2 = fX2; else x2 = fX1 + x2*dx;
         y1 = linel->GetY1();   if (y1 == 0) y1 = ytext; else y1 = fY1 + y1*dy;
         y2 = linel->GetY2();   if (y2 == 0) y2 = ytext; else y2 = fY1 + y2*dy;
         linel->PaintLine(x1,y1,x2,y2);
         continue;
      }
   // Next primitive is a box
      if (line->IsA() == TBox::Class()) {
         lineb = (TBox*)line;
         x1 = lineb->GetX1();
         if (x1) x1 = fX1 + x1*dx;
         else    x1 = fX1 + gPad->PixeltoX(1) - gPad->PixeltoX(0);
         x2 = lineb->GetX2();
         if (x2) x2 = fX1 + x2*dx;
         else    x2 = fX2;
         y1 = lineb->GetY1();   if (y1 == 0) y1 = ytext; else y1 = fY1 + y1*dy;
         y2 = lineb->GetY2();   if (y2 == 0) y2 = ytext; else y2 = fY1 + y2*dy;
         lineb->PaintBox(x1,y1,x2,y2);
         continue;
      }
   // Next primitive is a text
      if (line->IsA() == TText::Class()) {
         linet = (TText*)line;
         ytext -= yspace;
         Double_t xl    = linet->GetX();
         Double_t yl    = linet->GetY();
         Short_t talign = linet->GetTextAlign();
         Color_t tcolor = linet->GetTextColor();
         Style_t tfont  = linet->GetTextFont();
         Size_t  tsize  = linet->GetTextSize();
         if (talign == 0) linet->SetTextAlign(GetTextAlign());
         if (tcolor == 0) linet->SetTextColor(GetTextColor());
         if (tfont  == 0) linet->SetTextFont(GetTextFont());
         if (tsize  == 0) linet->SetTextSize(GetTextSize());
         if (xl > 0 && xl <1) {
            xtext = fX1 + xl*dx;
         } else {
            halign = linet->GetTextAlign()/10;
            if (halign == 1) xtext = fX1 + margin;
            if (halign == 2) xtext = 0.5*(fX1+fX2);
            if (halign == 3) xtext = fX2 - margin;
         }
         if (yl > 0 && yl <1) ytext = fY1 + yl*dy;
         linet->PaintText(xtext,ytext,linet->GetTitle());
         linet->SetTextAlign(talign);
         linet->SetTextColor(tcolor);
         linet->SetTextFont(tfont);
         linet->SetTextSize(tsize);
      }
   // Next primitive is a Latex text
      if (line->IsA() == TLatex::Class()) {
         latex = (TLatex*)line;
         ytext -= yspace;
         Double_t xl    = latex->GetX();
         Double_t yl    = latex->GetY();
         Short_t talign = latex->GetTextAlign();
         Color_t tcolor = latex->GetTextColor();
         Style_t tfont  = latex->GetTextFont();
         Size_t  tsize  = latex->GetTextSize();
         if (talign == 0) latex->SetTextAlign(GetTextAlign());
         if (tcolor == 0) latex->SetTextColor(GetTextColor());
         if (tfont  == 0) latex->SetTextFont(GetTextFont());
         if (tsize  == 0) latex->SetTextSize(GetTextSize());
         if (xl > 0 && xl <1) {
            xtext = fX1 + xl*dx;
         } else {
            halign = latex->GetTextAlign()/10;
            if (halign == 1) xtext = fX1 + margin;
            if (halign == 2) xtext = 0.5*(fX1+fX2);
            if (halign == 3) xtext = fX2 - margin;
         }
         if (yl > 0 && yl <1) ytext = fY1 + yl*dy;
         latex->PaintLatex(xtext,ytext,latex->GetTextAngle(),
                           latex->GetTextSize(),
                           latex->GetTitle());
         latex->SetTextAlign(talign);
         latex->SetTextColor(tcolor);
         latex->SetTextFont(tfont);
         latex->SetTextSize(tsize);
         latex->SetX(xl);  // PaintLatex modifies fX and fY
         latex->SetY(yl);
      }
   }

   SetTextSize(textsave);

   // if a label create & paint a pavetext title
   if (fLabel.Length() > 0) {
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
void TPaveText::Print(Option_t *option) const
{
   // Dump this pavetext with its attributes.

   TPave::Print(option);
   if (fLines) fLines->Print();
}


//______________________________________________________________________________
void TPaveText::ReadFile(const char *filename, Option_t *option, Int_t nlines, Int_t fromline)
{
   // Read lines of filename in this pavetext.
   //
   //  Read from line number fromline a total of nlines
   //
   //  Note that this function changes the default text alignment to left/center

   Int_t ival;
   Float_t val;
   TText *lastline = 0;
   TString opt = option;
   if (!opt.Contains("+")) {
      Clear();
      fLongest = 0;
   }
   SetTextAlign(12);
   // Get file name
   Int_t nch = strlen(filename);
   if (nch == 0) return;

   char *fname = StrDup(filename);
   if (fname[nch-1] == ';') { nch--; fname[nch]=0;}

   std::ifstream file(fname,std::ios::in);
   if (!file.good()) {
      Error("ReadFile", "illegal file name");
      delete [] fname;
      return;
   }

   const int linesize = 255;
   char currentline[linesize];
   char *ss, *sclose, *s= 0;

   Int_t kline = 0;
   while (1) {
      file.getline(currentline,linesize);
      if (file.eof())break;
      if (kline >= fromline && kline < fromline+nlines) {
         s = currentline;
         if (strstr(s,"+SetText")) {
            ss = s+8;
            sclose = strstr(ss,")");
            if (!sclose) continue;
            *sclose = 0;
            lastline = (TText*)fLines->Last();
            if (!lastline) continue;
            if (strstr(ss,"Color(")) {
               sscanf(ss+6,"%d",&ival);
               lastline->SetTextColor(ival);
               continue;
            }
            if (strstr(ss,"Align(")) {
               sscanf(ss+6,"%d",&ival);
               lastline->SetTextAlign(ival);
               continue;
            }
            if (strstr(ss,"Font(")) {
               sscanf(ss+5,"%d",&ival);
               lastline->SetTextFont(ival);
               continue;
            }
            if (strstr(ss,"Size(")) {
               sscanf(ss+5,"%f",&val);
               lastline->SetTextSize(val);
               continue;
            }
            if (strstr(ss,"Angle(")) {
               sscanf(ss+6,"%f",&val);
               lastline->SetTextAngle(val);
               continue;
            }
         }
         AddText(s);
      }
      kline++;
   }
   file.close();
   delete [] fname;
}


//______________________________________________________________________________
void TPaveText::SaveLines(std::ostream &out, const char *name)
{
   // Save lines of this pavetext as C++ statements on output stream out

   if (!fLines) return;
   Int_t nlines = GetSize();
   if (nlines == 0) return;

   // Iterate over all lines
   char quote = '"';
   TObject *line;
   TText *linet;
   TLatex *latex;
   TLine *linel;
   TBox  *lineb;
   TIter next(fLines);
   while ((line = (TObject*) next())) {
   // Next primitive is a line
      if (line->IsA() == TLine::Class()) {
         linel = (TLine*)line;
         if (gROOT->ClassSaved(TLine::Class())) {
            out<<"   ";
         } else {
            out<<"   TLine *";
         }
         out<<"line = "<<name<<"->AddLine("
            <<linel->GetX1()<<","<<linel->GetY1()<<","<<linel->GetX2()<<","<<linel->GetY2()<<");"<<std::endl;
         if (linel->GetLineColor() != 1) {
            if (linel->GetLineColor() > 228) {
               TColor::SaveColor(out, linel->GetLineColor());
               out<<"   line->SetLineColor(ci);" << std::endl;
            } else
               out<<"   line->SetLineColor("<<linel->GetLineColor()<<");"<<std::endl;
         }
         if (linel->GetLineStyle() != 1) {
            out<<"   line->SetLineStyle("<<linel->GetLineStyle()<<");"<<std::endl;
         }
         if (linel->GetLineWidth() != 1) {
            out<<"   line->SetLineWidth("<<linel->GetLineWidth()<<");"<<std::endl;
         }
         continue;
      }
   // Next primitive is a box
      if (line->IsA() == TBox::Class()) {
         lineb = (TBox*)line;
         if (gROOT->ClassSaved(TBox::Class())) {
            out<<"   ";
         } else {
            out<<"   TBox *";
         }
         out<<"box = "<<name<<"->AddBox("
            <<lineb->GetX1()<<","<<lineb->GetY1()<<","<<lineb->GetX2()<<","<<lineb->GetY2()<<");"<<std::endl;
         if (lineb->GetFillColor() != 18) {
            if (lineb->GetFillColor() > 228) {
               TColor::SaveColor(out, lineb->GetFillColor());
               out<<"   box->SetFillColor(ci);" << std::endl;
            } else
               out<<"   box->SetFillColor("<<lineb->GetFillColor()<<");"<<std::endl;
         }
         if (lineb->GetFillStyle() != 1001) {
            out<<"   box->SetFillStyle("<<lineb->GetFillStyle()<<");"<<std::endl;
         }
         if (lineb->GetLineColor() != 1) {
            if (lineb->GetLineColor() > 228) {
               TColor::SaveColor(out, lineb->GetLineColor());
               out<<"   box->SetLineColor(ci);" << std::endl;
            } else
               out<<"   box->SetLineColor("<<lineb->GetLineColor()<<");"<<std::endl;
         }
         if (lineb->GetLineStyle() != 1) {
            out<<"   box->SetLineStyle("<<lineb->GetLineStyle()<<");"<<std::endl;
         }
         if (lineb->GetLineWidth() != 1) {
            out<<"   box->SetLineWidth("<<lineb->GetLineWidth()<<");"<<std::endl;
         }
         continue;
      }
   // Next primitive is a text
      if (line->IsA() == TText::Class()) {
         linet = (TText*)line;
         if (gROOT->ClassSaved(TText::Class())) {
            out<<"   ";
         } else {
            out<<"   TText *";
         }
         if (!linet->GetX() && !linet->GetY()) {
            TString s = linet->GetTitle();
            s.ReplaceAll("\"","\\\"");
            out<<"AText = "<<name<<"->AddText("
               <<quote<<s.Data()<<quote<<");"<<std::endl;
         } else {
            out<<"AText = "<<name<<"->AddText("
               <<linet->GetX()<<","<<linet->GetY()<<","<<quote<<linet->GetTitle()<<quote<<");"<<std::endl;
         }
         if (linet->GetTextColor()) {
            if (linet->GetTextColor() > 228) {
               TColor::SaveColor(out, linet->GetTextColor());
               out<<"   AText->SetTextColor(ci);" << std::endl;
            } else
               out<<"   AText->SetTextColor("<<linet->GetTextColor()<<");"<<std::endl;
         }
         if (linet->GetTextFont()) {
            out<<"   AText->SetTextFont("<<linet->GetTextFont()<<");"<<std::endl;
         }
         if (linet->GetTextSize()) {
            out<<"   AText->SetTextSize("<<linet->GetTextSize()<<");"<<std::endl;
         }
         if (linet->GetTextAngle() != GetTextAngle()) {
            out<<"   AText->SetTextAngle("<<linet->GetTextAngle()<<");"<<std::endl;
         }
         if (linet->GetTextAlign()) {
            out<<"   AText->SetTextAlign("<<linet->GetTextAlign()<<");"<<std::endl;
         }
      }
   // Next primitive is a Latex text
      if (line->IsA() == TLatex::Class()) {
         latex = (TLatex*)line;
         if (gROOT->ClassSaved(TLatex::Class())) {
            out<<"   ";
         } else {
            out<<"   TText *";
         }
         if (!latex->GetX() && !latex->GetY()) {
            TString sl = latex->GetTitle();
            sl.ReplaceAll("\"","\\\"");
            out<<"AText = "<<name<<"->AddText("
               <<quote<<sl.Data()<<quote<<");"<<std::endl;
         } else {
            out<<"AText = "<<name<<"->AddText("
               <<latex->GetX()<<","<<latex->GetY()<<","<<quote<<latex->GetTitle()<<quote<<");"<<std::endl;
         }
         if (latex->GetTextColor()) {
            if (latex->GetTextColor() > 228) {
               TColor::SaveColor(out, latex->GetTextColor());
               out<<"   AText->SetTextColor(ci);" << std::endl;
            } else
               out<<"   AText->SetTextColor("<<latex->GetTextColor()<<");"<<std::endl;
         }
         if (latex->GetTextFont()) {
            out<<"   AText->SetTextFont("<<latex->GetTextFont()<<");"<<std::endl;
         }
         if (latex->GetTextSize()) {
            out<<"   AText->SetTextSize("<<latex->GetTextSize()<<");"<<std::endl;
         }
         if (latex->GetTextAngle() != GetTextAngle()) {
            out<<"   AText->SetTextAngle("<<latex->GetTextAngle()<<");"<<std::endl;
         }
         if (latex->GetTextAlign()) {
            out<<"   AText->SetTextAlign("<<latex->GetTextAlign()<<");"<<std::endl;
         }
      }
   }
}


//______________________________________________________________________________
void TPaveText::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<std::endl;
   if (gROOT->ClassSaved(TPaveText::Class())) {
      out<<"   ";
   } else {
      out<<"   "<<ClassName()<<" *";
   }
   if (fOption.Contains("NDC")) {
      out<<"pt = new "<<ClassName()<<"("<<fX1NDC<<","<<fY1NDC<<","<<fX2NDC<<","<<fY2NDC
      <<","<<quote<<fOption<<quote<<");"<<std::endl;
   } else {
      out<<"pt = new "<<ClassName()<<"("<<gPad->PadtoX(fX1)<<","<<gPad->PadtoY(fY1)<<","<<gPad->PadtoX(fX2)<<","<<gPad->PadtoY(fY2)
      <<","<<quote<<fOption<<quote<<");"<<std::endl;
   }
   if (strcmp(GetName(),"TPave")) {
      out<<"   pt->SetName("<<quote<<GetName()<<quote<<");"<<std::endl;
   }
   if (fLabel.Length() > 0) {
      out<<"   pt->SetLabel("<<quote<<fLabel<<quote<<");"<<std::endl;
   }
   if (fBorderSize != 4) {
      out<<"   pt->SetBorderSize("<<fBorderSize<<");"<<std::endl;
   }
   SaveFillAttributes(out,"pt",19,1001);
   SaveLineAttributes(out,"pt",1,1,1);
   SaveTextAttributes(out,"pt",22,0,1,62,0);
   SaveLines(out,"pt");
   out<<"   pt->Draw();"<<std::endl;
}


//______________________________________________________________________________
void TPaveText::SetAllWith(const char *text, Option_t *option, Double_t value)
{
   // Set attribute option for all lines containing string text.
   //
   // Possible options are all the AttText attributes
   //       Align, Color, Font, Size and Angle

   TString opt=option;
   opt.ToLower();
   TText *line;
   TIter next(fLines);
   while ((line = (TText*) next())) {
      if (strstr(line->GetTitle(),text)) {
         if (opt == "align") line->SetTextAlign(Int_t(value));
         if (opt == "color") line->SetTextColor(Int_t(value));
         if (opt == "font")  line->SetTextFont(Int_t(value));
         if (opt == "size")  line->SetTextSize(value);
         if (opt == "angle") line->SetTextAngle(value);
      }
   }
}


//______________________________________________________________________________
void TPaveText::Streamer(TBuffer &R__b)
{
   // Stream an object of class TPaveText.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         R__b.ReadClassBuffer(TPaveText::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TPave::Streamer(R__b);
      TAttText::Streamer(R__b);
      if (R__v > 1) fLabel.Streamer(R__b);
      R__b >> fLongest;
      R__b >> fMargin;
      R__b >> fLines;
      R__b.CheckByteCount(R__s, R__c, TPaveText::IsA());
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TPaveText::Class(),this);
   }
}

//______________________________________________________________________________
void TPaveText::UseCurrentStyle()
{
   // Replace current attributes by current style.

   if (gStyle->IsReading()) {
      SetTextFont(gStyle->GetTextFont());
      SetTextSize(gStyle->GetTextSize());
      SetTextColor(gStyle->GetTextColor());
   } else {
      gStyle->SetTextColor(GetTextColor());
      gStyle->SetTextFont(GetTextFont());
      gStyle->SetTextSize(GetTextSize());
   }
}
