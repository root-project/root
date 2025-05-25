// @(#)root/gpad:$Id$
// Author: Rene Brun   01/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TButton.h"
#include "TCanvas.h"
#include "TLatex.h"

#include <cstring>
#include <iostream>

ClassImp(TButton);

/** \class TButton
\ingroup gpad

A TButton object is a user interface object.

A TButton has a name and an associated action.
When the button is clicked with the left mouse button, the corresponding
action is executed.

A TButton can be created by direct invocation of the constructors
or via the graphics editor.

The action can be set via TButton::SetMethod.
The action can be any command. Examples of actions:
  - "34+78" When the button is clicked, the result of addition is printed.
  - ".x macro.C" . Clicking the button executes the macro macro.C
The action can be modified at any time via TButton::SetMethod.

To modify the layout/size/contents of one or several buttons
in a canvas, you must set the canvas editable via TCanvas::SetEditable.
By default a TCanvas is editable.
By default a TDialogCanvas is not editable.
TButtons are in general placed in a TDialogCanvas.

A TButton being a TPad, one can draw graphics primitives in it
when the TCanvas/TDialogCanvas is editable.

Example of a macro creating a dialog canvas with buttons:
~~~ {.cpp}
void but() {
//   example of a dialog canvas with a few buttons

   TDialogCanvas *dialog = new TDialogCanvas("dialog","",200,300);

// Create first button. Clicking on this button will execute 34+56
   TButton *but1 = new TButton("button1","34+56",.05,.8,.45,.88);
   but1->Draw();

// Create second button. Clicking on this button will create a new canvas
   TButton *but2 = new TButton("canvas","c2 = new TCanvas(\"c2\")",.55,.8,.95,.88);
   but2->Draw();

// Create third button. Clicking on this button will invoke the browser
   but3 = new TButton("Browser","br = new TBrowser(\"br\")",0.25,0.54,0.75,0.64);
   but3->SetFillColor(42);
   but3->Draw();

// Create last button with no name. Instead a graph is draw inside the button
// Clicking on this button will invoke the macro $ROOTSYS/tutorials/visualisation/graphs/gr001_simple.C
   button = new TButton("",".x tutorials/visualisation/graphs/gr001_simple.C",0.15,0.15,0.85,0.38);
   button->SetFillColor(42);
   button->Draw();
   button->SetEditable(kTRUE);
   button->cd();

   Double_t x[8] = {0.08,0.21,0.34,0.48,0.61,0.7,0.81,0.92};
   Double_t y[8] = {0.2,0.65,0.4,0.34,0.24,0.43,0.75,0.52};
   TGraph *graph = new TGraph(8,x,y);
   graph->SetMarkerColor(4);
   graph->SetMarkerStyle(21);
   graph->Draw("lp");

   dialog->cd();
}
~~~
Executing the macro above produces the following dialog canvas:

\image html gpad_dialogbuttons.png
*/

////////////////////////////////////////////////////////////////////////////////
/// Button default constructor.

TButton::TButton(): TPad()
{
   fFraming = kFALSE;
   fMethod  = "";
   fLogx    = kFALSE;
   fLogy    = kFALSE;
   SetEditable(kFALSE);
   fFocused = kFALSE;
   for (UChar_t n = 0; n < 128; ++n)
      fValidPattern[n] = n;
}

////////////////////////////////////////////////////////////////////////////////
/// Button normal constructor.
///
///   Note that the button coordinates x1,y1,x2,y2 are always in the range [0,1]

TButton::TButton(const char *title, const char *method, Double_t x1, Double_t y1,Double_t x2, Double_t  y2)
           :TPad("button",title,x1,y1,x2,y2,18,2,1), TAttText(22,0,1,61,0.65)
{
   fFraming = kFALSE;
   SetBit(kCanDelete);
   fModified = kTRUE;
   fMethod = method;
   if (title && strlen(title)) {
      TLatex *text = new TLatex(0.5 * (fX1 + fX2), 0.5 * (fY1 + fY2), title);
      fPrimitives->Add(text);
   }
   fLogx    = 0;
   fLogy    = 0;
   SetEditable(kFALSE);
   fFocused = kFALSE;
   for (UChar_t n = 0; n < 128; ++n)
      fValidPattern[n] = n;
}

////////////////////////////////////////////////////////////////////////////////
/// Button default destructor.

TButton::~TButton()
{
   for (UChar_t n = 0; n < 128; ++n)
      fValidPattern[n] = 255 - n;
   if (fPrimitives)
      fPrimitives->Delete();
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this button with its current attributes.

void TButton::Draw(Option_t *option)
{
   if (fCanvas) AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
///  This member function is called when a Button object is clicked.

void TButton::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   //check case where pressing a button deletes itself
   if (ROOT::Detail::HasBeenDeleted(this)) return;

   if (IsEditable()) {
      TPad::ExecuteEvent(event,px,py);
      return;
   }

   auto cdpad = gROOT->GetSelectedPad();
   auto patt = fValidPattern;
   HideToolTip(event);

   switch (event) {

   case kMouseEnter:
      TPad::ExecuteEvent(event,px,py);
      break;

   case kButton1Down:
      SetBorderMode(-1);
      fFocused = kTRUE;
      Modified();
      Update();
      break;

   case kMouseMotion:

      break;

   case kButton1Motion:
      if (px<XtoAbsPixel(1) && px>XtoAbsPixel(0) &&
          py<YtoAbsPixel(0) && py>YtoAbsPixel(1)) {
         if (!fFocused) {
            SetBorderMode(-1);
            fFocused = kTRUE;
            Modified();
            GetCanvas()->Modified();
            Update();
         }
      } else if (fFocused) {
         SetBorderMode(1);
         fFocused = kFALSE;
         Modified();
         GetCanvas()->Modified();
         Update();
      }
      break;

   case kButton1Up:
      SetCursor(kWatch);
      if (fFocused) {
         if (cdpad) cdpad->cd();
         gROOT->ProcessLine(GetMethod());
      }
      //check case where pressing a button deletes itself
      if (ROOT::Detail::HasBeenDeleted(this))
         return;

      // extra check when simple one does not work
      for (UChar_t n = 0; n < 128; ++n)
         if (patt[n] != n)
            return;

      SetBorderMode(1);
      Modified();
      Update();
      SetCursor(kCross);
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this button with its current attributes.

void TButton::Paint(Option_t *option)
{
   if (!fCanvas) return;
   if (!fPrimitives) fPrimitives = new TList();
   TObject *obj = GetListOfPrimitives()->First();
   if (obj && obj->InheritsFrom(TLatex::Class())) {
      TLatex *text = (TLatex*)obj;
      text->SetTitle(GetTitle());
      text->SetTextSize(GetTextSize());
      text->SetTextFont(GetTextFont());
      text->SetTextAlign(GetTextAlign());
      text->SetTextColor(GetTextColor());
      text->SetTextAngle(GetTextAngle());
   }
   SetLogx(0);
   SetLogy(0);
   TPad::Paint(option);  //only called for Postscript print
}

////////////////////////////////////////////////////////////////////////////////
/// Paint is modified.

void TButton::PaintModified()
{
   if (!fCanvas) return;
   if (!fPrimitives) fPrimitives = new TList();
   TObject *obj = GetListOfPrimitives()->First();
   if (obj && obj->InheritsFrom(TLatex::Class())) {
      TLatex *text = (TLatex*)obj;
      text->SetTitle(GetTitle());
      text->SetTextSize(GetTextSize());
      text->SetTextFont(GetTextFont());
      text->SetTextAlign(GetTextAlign());
      text->SetTextColor(GetTextColor());
      text->SetTextAngle(GetTextAngle());
   }
   SetLogx(0);
   SetLogy(0);
   TPad::PaintModified();
}

////////////////////////////////////////////////////////////////////////////////
/// Set world coordinate system for the pad.

void TButton::Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2)
{
   TPad::Range(x1,y1,x2,y2);
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TButton::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   SavePrimitiveConstructor(out, Class(), "button",
                            TString::Format("\"%s\", \"%s\", %g, %g, %g, %g",
                                            TString(GetTitle()).ReplaceSpecialCppChars().Data(),
                                            TString(GetMethod()).ReplaceSpecialCppChars().Data(), fXlowNDC, fYlowNDC,
                                            fXlowNDC + fWNDC, fYlowNDC + fHNDC));

   SaveFillAttributes(out, "button", 0, 1001);
   SaveLineAttributes(out, "button", 1, 1, 1);
   SaveTextAttributes(out, "button", 22, 0, 1, 61, .65);

   if (GetBorderSize() != 2)
      out << "   button->SetBorderSize(" << GetBorderSize() << ");\n";
   if (GetBorderMode() != 1)
      out << "   button->SetBorderMode(" << GetBorderMode() << ");\n";

   if (GetFraming())
      out << "button->SetFraming();\n";
   if (IsEditable())
      out << "button->SetEditable(kTRUE);\n";

   out << "   button->Draw();\n";

   TIter next(GetListOfPrimitives());
   next(); // do not save first primitive which should be text

   Int_t nprim = 0;
   while (auto obj = next()) {
      if (nprim++ == 0)
         out << "   button->cd();\n";
      obj->SavePrimitive(out, next.GetOption());
   }

   if ((nprim > 0) && gPad)
      out << "   " << gPad->GetName() << "->cd();\n";
}

////////////////////////////////////////////////////////////////////////////////
/// if framing is set, button will be highlighted

void TButton::SetFraming(Bool_t f)
{
   fFraming = f;
   if (f) SetBit(kFraming);
   else   ResetBit(kFraming);
}
