// @(#)root/graf:$Id$
// Author: Rene Brun   20/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>

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
#include "TLine.h"

ClassImp(TPaveText);

/** \class TPaveText
\ingroup BasicGraphics

A Pave (see TPave) with text, lines or/and boxes inside.

Line (and boxes) are positioned in the pave using coordinates relative to
the pave (%).

The text lines are added in order using the AddText method. Also line separators
can be added, in order too, using the AddLine method.

AddText returns a TText corresponding to the line added to the pave. This
return value can be used to modify the text attributes.

Once the TPaveText is build the text of each line can be retrieved using
GetLine or GetLineWith as a TText wich is useful to modify the text attributes
of a line.

Example:
Begin_Macro(source)
../../../tutorials/visualisation/graphics/pavetext.C
End_Macro

GetListOfLines can also be used to retrieve all the lines in the TPaveText as
a TList:

Begin_Macro(source)
{
   TPaveText *t = new TPaveText(.05,.3,.95,.6);
   t->AddText("This line is blue"); ((TText*)t->GetListOfLines()->Last())->SetTextColor(kBlue);
   t->AddText("This line is red");  ((TText*)t->GetListOfLines()->Last())->SetTextColor(kRed);
   t->Draw();
}
End_Macro

*/

////////////////////////////////////////////////////////////////////////////////
/// pavetext default constructor.

TPaveText::TPaveText(): TPave(), TAttText()
{
   fLines   = nullptr;
   fMargin  = 0.05;
   fLongest = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// PaveText normal constructor.
///
/// A PaveText is a Pave with several lines of text
///
///  - option = "TR" Top and Right shadows are drawn.
///  - option = "TL" Top and Left shadows are drawn.
///  - option = "BR" Bottom and Right shadows are drawn.
///  - option = "BL" Bottom and Left shadows are drawn.
///
///  If none of these four above options is specified the default the
///  option "BR" will be used to draw the border. To produces a pave
///  without any border it is enough to specify the option "NB" (no border).
///  If you want to remove the border or shadow of an already existing TPaveText,
///  then use the function TPave::SetBorderSize.
///
///  - option = "NDC" x1,y1,x2,y2 are given in NDC
///  - option = "ARC" corners are rounded
///
/// In case of option "ARC", the corner radius is specified
/// via TPave::SetCornerRadius(rad) where rad is given in percent
/// of the pave height (default value is 0.2).
///
/// The individual text items are entered via AddText
/// By default, text items inherits from the default pavetext AttText.
/// A title can be added later to this pavetext via TPaveText::SetLabel.

TPaveText::TPaveText(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, Option_t *option)
           :TPave(x1,y1,x2,y2,4,option), TAttText(22,0,gStyle->GetTextColor(),gStyle->GetTextFont(),0)
{
   fLines   = new TList;
   fMargin  = 0.05;
   fLongest = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// pavetext default destructor.

TPaveText::~TPaveText()
{
   if (ROOT::Detail::HasBeenDeleted(this)) return;
   if (fLines) fLines->Delete();
   delete fLines;
   fLines = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// pavetext copy constructor.

TPaveText::TPaveText(const TPaveText &pavetext) : TPave(pavetext), TAttText(pavetext)
{
   fLabel = pavetext.fLabel;
   fLongest = pavetext.fLongest;
   fMargin = pavetext.fMargin;
   if (pavetext.fLines)
      fLines = (TList *) pavetext.fLines->Clone();
}

////////////////////////////////////////////////////////////////////////////////
///assignment operator

TPaveText& TPaveText::operator=(const TPaveText& pt)
{
   if(this != &pt) {
      TPave::operator=(pt);
      TAttText::operator=(pt);
      fLabel = pt.fLabel;
      fLongest = pt.fLongest;
      fMargin = pt.fMargin;
      if (fLines) {
         fLines->Delete();
         delete fLines;
         fLines = nullptr;
      }
      if (pt.fLines)
         fLines = (TList *)pt.fLines->Clone();
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new graphics box to this pavetext.

TBox *TPaveText::AddBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (!gPad->IsEditable())
      return nullptr;
   TBox *newbox = new TBox(x1,y1,x2,y2);

   if (!fLines) fLines = new TList;
   fLines->Add(newbox);
   return newbox;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new graphics line to this pavetext.

TLine *TPaveText::AddLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   if (!gPad->IsEditable())
      return nullptr;
   TLine *newline = new TLine(x1,y1,x2,y2);

   if (!fLines) fLines = new TList;
   fLines->Add(newline);
   return newline;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new Text line to this pavetext at given coordinates.

TText *TPaveText::AddText(Double_t x1, Double_t y1, const char *text)
{
   TLatex *newtext = new TLatex(x1,y1,text);
   newtext->SetTextAlign(0);
   newtext->SetTextColor(0);
   newtext->SetTextFont(0);
   newtext->SetTextSize(0);
   Int_t nch = text ? strlen(text) : 0;
   if (nch > fLongest) fLongest = nch;

   if (!fLines) fLines = new TList;
   fLines->Add(newtext);
   return newtext;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new Text line to this pavetext.

TText *TPaveText::AddText(const char *text)
{
   return AddText(0,0,text);
}

////////////////////////////////////////////////////////////////////////////////
/// Clear all lines in this pavetext.

void TPaveText::Clear(Option_t *)
{
   if (!fLines) return;
   fLines->Delete();
   fLongest = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete text at the mouse position.

void TPaveText::DeleteText()
{
   if (!gPad->IsEditable()) return;
   if (!fLines) return;
   Double_t ymouse, yobj;
   TObject *obj = GetObject(ymouse, yobj);             //get object pointed by the mouse
   if (!obj) return;
   if (!obj->InheritsFrom(TText::Class())) return;
   fLines->Remove(obj);
   delete obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this pavetext with its current attributes.

void TPaveText::Draw(Option_t *option)
{
   Option_t *opt;
   if (option && strlen(option)) opt = option;
   else                          opt = GetOption();

   AppendPad(opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw lines in filename in this pavetext.

void TPaveText::DrawFile(const char *filename, Option_t *option)
{
   ReadFile(filename);

   AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Edit text at the mouse position.

void TPaveText::EditText()
{
   if (!gPad->IsEditable()) return;
   Double_t ymouse, yobj;
   TObject *obj = GetObject(ymouse, yobj);             //get object pointed by the mouse
   if (!obj) return;
   if (!obj->InheritsFrom(TText::Class())) return;
   TText *text = (TText*)obj;
   gROOT->SetSelectedPrimitive(text);
   gROOT->ProcessLine(TString::Format("((TCanvas*)0x%zx)->SetSelected((TObject*)0x%zx)",
                           (size_t)gPad->GetCanvas(), (size_t)text));
   gROOT->ProcessLine(TString::Format("((TCanvas*)0x%zx)->Selected((TVirtualPad*)0x%zx,(TObject*)0x%zx,1)",
                           (size_t)gPad->GetCanvas(), (size_t)gPad, (size_t)text));
   text->SetTextAttributes();
}

////////////////////////////////////////////////////////////////////////////////
/// Get Pointer to line number in this pavetext.
/// Ignore any TLine or TBox, they are not accounted

TText *TPaveText::GetLine(Int_t number) const
{
   TIter next(fLines);
   Int_t nlines = 0;
   while (auto obj = next()) {
      if (!obj->InheritsFrom(TText::Class()))
         continue;

      if (nlines++ == number)
         return (TText *) obj;
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get Pointer to first containing string text in this pavetext.
/// Ignore any TLine or TBox, they are not accounted

TText *TPaveText::GetLineWith(const char *text) const
{
   if (!text)
      return nullptr;
   TIter next(fLines);
   while (auto obj = next()) {
      if (obj->InheritsFrom(TText::Class()) && strstr(obj->GetTitle(), text))
         return (TText *) obj;
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get object pointed by the mouse in this pavetext.

TObject *TPaveText::GetObject(Double_t &ymouse, Double_t &yobj) const
{
   if (!fLines) return nullptr;
   Int_t nlines = GetSize();
   if (nlines == 0) return nullptr;

   // Evaluate text size as a function of the number of lines

   ymouse   = gPad->AbsPixeltoY(gPad->GetEventY());
   Double_t yspace   = (fY2 - fY1)/Double_t(nlines);
   Double_t y1,y,dy;
   Double_t ytext = fY2 + 0.5*yspace;
   Int_t valign;

   // Iterate over all lines
   // Copy pavetext attributes to line attributes if line attributes not set
   dy = fY2 - fY1;
   TIter next(fLines);
   while (auto line = next()) {
      // Next primitive is a line
      if (line->IsA() == TLine::Class()) {
         auto linel = (TLine *)line;
         y1 = linel->GetY1();   if (y1 == 0) y1 = ytext; else y1 = fY1 + y1*dy;
         if (TMath::Abs(y1-ymouse) < 0.2*yspace) {yobj = y1; return line;}
         continue;
      }
      // Next primitive is a box
      if (line->IsA() == TBox::Class()) {
         auto lineb = (TBox *)line;
         y1 = lineb->GetY1();
         if (y1 == 0) y1 = ytext; else y1 = fY1 + y1*dy;
         if (TMath::Abs(y1-ymouse) < 0.4*yspace) {yobj = y1; return line;}
         continue;
      }
      // Next primitive is a text
      if (line->InheritsFrom(TText::Class())) {
         auto linet = (TText *)line;
         ytext -= yspace;
         Double_t yl     = linet->GetY();
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
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///  return number of text lines (ignoring TLine, etc)

Int_t TPaveText::GetSize() const
{
   Int_t nlines = 0;
   TIter next(fLines);
   while (auto line = next()) {
      if (line->InheritsFrom(TText::Class())) nlines++;
   }
   return nlines;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a new line at the mouse position.

void TPaveText::InsertLine()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Add a new Text line at the mouse position.

void TPaveText::InsertText(const char *text)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Paint this pavetext with its current attributes.

void TPaveText::Paint(Option_t *option)
{
   // Draw the pave
   TPave::ConvertNDCtoPad();
   TPave::PaintPave(fX1,fY1,fX2,fY2,GetBorderSize(),option);
   PaintPrimitives(kPaveText);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint list of primitives in this pavetext.

void TPaveText::PaintPrimitives(Int_t mode)
{
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
      TString opt = GetDrawOption(); opt.ReplaceAll("NDC", "");
      TPaveLabel title(x1,y1,x2,y2,fLabel.Data(),opt.Data());
      title.SetFillColor(GetFillColor());
      title.SetTextColor(GetTextColor());
      title.SetTextFont(GetTextFont());
      title.Paint();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Dump this pavetext with its attributes.

void TPaveText::Print(Option_t *option) const
{
   TPave::Print(option);
   if (fLines) fLines->Print();
}

////////////////////////////////////////////////////////////////////////////////
/// Read lines of filename in this pavetext.
///
///  Read from line number fromline a total of nlines
///
///  Note that this function changes the default text alignment to left/center

void TPaveText::ReadFile(const char *filename, Option_t *option, Int_t nlines, Int_t fromline)
{
   Int_t ival;
   Float_t val;
   TString opt = option;
   if (!opt.Contains("+")) {
      Clear();
      fLongest = 0;
   }
   SetTextAlign(12);
   // Get file name
   TString fname = filename;
   if (fname.EndsWith(";"))
      fname.Resize(fname.Length() - 1);
   if (fname.Length() == 0)
      return;

   std::ifstream file(fname.Data(),std::ios::in);
   if (!file.good()) {
      Error("ReadFile", "illegal file name %s", fname.Data());
      return;
   }

   const int linesize = 255;
   char currentline[linesize];
   char *ss, *sclose, *s = nullptr;

   Int_t kline = 0;
   while (true) {
      file.getline(currentline,linesize);
      if (file.eof())break;
      if (kline >= fromline && kline < fromline+nlines) {
         s = currentline;
         if (strstr(s,"+SetText")) {
            ss = s+8;
            sclose = strstr(ss,")");
            if (!sclose) continue;
            *sclose = 0;
            TText *lastline = (TText*)fLines->Last();
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
}

////////////////////////////////////////////////////////////////////////////////
/// Save lines of this pavetext as C++ statements on output stream out

void TPaveText::SaveLines(std::ostream &out, const char *name, Bool_t)
{
   if (!fLines) return;
   Int_t nlines = GetSize();
   if (nlines == 0) return;

   if (!name || !*name)
      name = "pt";

   static int linecnt = 0;

   // Iterate over all lines
   TIter next(fLines);

   while (auto line = next()) {
      if (line->IsA() == TLine::Class()) {
         // Next primitive is a line
         auto linel = static_cast<TLine *>(line);
         auto line_name = TString::Format("%s_line%d", name, linecnt++);
         out << "   TLine *" << line_name << " = " << name << "->AddLine(" << linel->GetX1() << "," << linel->GetY1()
             << "," << linel->GetX2() << "," << linel->GetY2() << ");" << std::endl;
         linel->SaveLineAttributes(out, line_name.Data(), 1, 1, 1);
      } else if (line->IsA() == TBox::Class()) {
         // Next primitive is a box
         auto lineb = static_cast<TBox *>(line);
         auto box_name = TString::Format("%s_box%d", name, linecnt++);
         out << "   TBox *" << box_name << " = " << name << "->AddBox(" << lineb->GetX1() << "," << lineb->GetY1()
             << "," << lineb->GetX2() << "," << lineb->GetY2() << ");" << std::endl;
         lineb->SaveFillAttributes(out, box_name.Data(), 18, 1001);
         lineb->SaveLineAttributes(out, box_name.Data(), 1, 1, 1);
      } else if (line->IsA() == TText::Class() || line->IsA() == TLatex::Class()) {
         // Next primitive is a text
         auto linet = static_cast<TText *>(line);
         auto text_name = TString::Format("%s_text%d", name, linecnt++);

         TString s = linet->GetTitle();
         s.ReplaceSpecialCppChars();

         out << "   TText *" << text_name << " = ";

         if (!linet->GetX() && !linet->GetY())
            out << name << "->AddText(\"" << s << "\");" << std::endl;
         else
            out << name << "->AddText(" << linet->GetX() << ", " << linet->GetY() << ", \"" << s << "\");" << std::endl;

         linet->SaveTextAttributes(out, text_name.Data(), 0, GetTextAngle(), 0, 0, 0);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TPaveText::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(out, Class(), "pt", GetSavePaveArgs());

   if (strcmp(GetName(), "TPave"))
      out << "   pt->SetName(\"" << GetName() << "\");\n";
   if (fLabel.Length() > 0)
      out << "   pt->SetLabel(\"" << TString(fLabel).ReplaceSpecialCppChars() << "\");\n";
   if (fBorderSize != 4)
      out << "   pt->SetBorderSize(" << fBorderSize << ");\n";
   SaveFillAttributes(out, "pt", 19, 1001);
   SaveLineAttributes(out, "pt", 1, 1, 1);
   SaveTextAttributes(out, "pt", 22, 0, 1, 62, 0);
   SaveLines(out, "pt", kTRUE);
   SavePrimitiveDraw(out, "pt", option);
}

////////////////////////////////////////////////////////////////////////////////
/// Set attribute option for all lines containing string text.
///
/// Possible options are all the AttText attributes
///       Align, Color, Font, Size and Angle

void TPaveText::SetAllWith(const char *text, Option_t *option, Double_t value)
{
   TString opt = option;
   opt.ToLower();
   TIter next(fLines);
   while (auto obj = next()) {
      auto line = dynamic_cast<TText *> (obj);
      if (line && strstr(line->GetTitle(),text)) {
         if (opt == "align") line->SetTextAlign(Int_t(value));
         if (opt == "color") line->SetTextColor(Int_t(value));
         if (opt == "font")  line->SetTextFont(Int_t(value));
         if (opt == "size")  line->SetTextSize(value);
         if (opt == "angle") line->SetTextAngle(value);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TPaveText.

void TPaveText::Streamer(TBuffer &R__b)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Replace current attributes by current style.

void TPaveText::UseCurrentStyle()
{
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
