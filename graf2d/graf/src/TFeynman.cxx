// @(#)root/feynman:$Id$
// Author: Advait Dhingra   12/04/2021

/** \class TFeynman
    \ingroup feynman
TFeynman is a class that makes it easier to make
good-looking Feynman Diagrams using ROOT components
like TArc and TArrow.
### Decleration / Access to the components
TFeynman is initialized with the width and the height
of the Canvas that you would like.
~~~
  TFeynman *f = new TFeynman(300, 600);
~~~
You can access the particle classes using an arrow pointer.
This example plots the feynman.C diagram in the tutorials:
~~~
  f->Lepton(10, 10, 30, 30, 7, 6, "e", true)->Draw();
  f->Lepton(10, 50, 30, 30, 5, 55, "e", false)->Draw();
  f->CurvedPhoton(30, 30, 12.5*sqrt(2), 135, 225, 7, 30)->Draw();
  f->Photon(30, 30, 55, 30, 42.5, 37.7)->Draw();
  f->QuarkAntiQuark(70, 30, 15, 55, 45, "q")->Draw();
  f->Gluon(70, 45, 70, 15, 77.5, 30)->Draw();
  f->WeakBoson(85, 30, 110, 30, 100, 37.5, "Z^{0}")->Draw();
  f->Quark(110, 30, 130, 50, 135, 55, "q", true)->Draw();
  f->Quark(110, 30, 130, 10, 135, 6, "q", false)->Draw();
  f->CurvedGluon(110, 30, 12.5*sqrt(2), 315, 45, 135, 30)->Draw();
~~~

*/

#include <cstdio>
#include <iostream>

#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TMarker.h"
#include "../inc/TFeynman.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TROOT.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TH1.h"
#include "THStack.h"

ClassImp(TFeynman);


TFeynman::TFeynman(Double_t canvasWidth, Double_t canvasHeight){
				TCanvas *c1 = new TCanvas("c1", "c1", 10,10, canvasWidth, canvasHeight);
   			c1->Range(0, 0, 140, 60);
				gStyle->SetLineWidth(2);
        fPrimitives = new TList();
		}

TFeynmanEntry* TFeynman::AddEntry(const TObject *particle) {
   TFeynmanEntry *newEntry = new TFeynmanEntry(particle);
   fPrimitives->Add((TObject*)newEntry);
	 cout << fPrimitives << endl;
   return newEntry;
}

void TFeynman::Draw(Option_t* option="") {
	AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a Quark
/// \param[in] x1 starting point in x direction
/// \param[in] y1 starting point in y direction
/// \param[in] x2 stopping point in x direction
/// \param[in] y2 stopping point in y direction
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
/// \param[in] quarkName name of the Quark (u, d, c, s, t, b or q)
/// \param[in] isMatter is the particle matter (true) or antimatter (false)
///
/// The TArrow is returned and the label is simply drawn.
TArrow *TFeynman::Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX,
                        Double_t labelPositionY, const char *quarkName, bool isMatter)
{
   TArrow *q;

   q = new TArrow(x1, y1, x2, y2, 0.02, "->-");

   const char *usedQuarkName;

   if (isMatter == true) {
      if (quarkName == std::string("q")) {
         usedQuarkName = "q";
      } else if (quarkName == std::string("u")) {
         usedQuarkName = "u";
      } else if (quarkName == std::string("d")) {
         usedQuarkName = "d";
      } else if (quarkName == std::string("c")) {
         usedQuarkName = "c";
      } else if (quarkName == std::string("s")) {
         usedQuarkName = "s";
      } else if (quarkName == std::string("t")) {
         usedQuarkName = "t";
      } else if (quarkName == std::string("b")) {
         usedQuarkName = "b";
      } else {
         usedQuarkName = "q";
      }
   }

   if (isMatter == false) {
      if (quarkName == std::string("q")) {
         usedQuarkName = "#bar{q}";
      } else if (quarkName == std::string("u")) {
         usedQuarkName = "#bar{u}";
      } else if (quarkName == std::string("d")) {
         usedQuarkName = "#bar{d}";
      } else if (quarkName == std::string("c")) {
         usedQuarkName = "#bar{c}";
      } else if (quarkName == std::string("s")) {
         usedQuarkName = "#bar{s}";
      } else if (quarkName == std::string("t")) {
         usedQuarkName = "#bar{t}";
      } else if (quarkName == std::string("b")) {
         usedQuarkName = "#bar{b}";
      } else {
         usedQuarkName = "#bar{b}";
      }
   }
   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, usedQuarkName);

   return q;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Quark Antiquark pair
/// \param[in] x1 x coordinate of arc centre
/// \param[in] y1 y coordinate of arc centre
/// \param[in] rad radius of the arc
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
/// \param[in] quarkName name of the quark (not the antiquark)
///
/// The TArc is returned and the labels are drawn automatically
TArc *TFeynman::QuarkAntiQuark(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY,
                               const char *quarkName)
{
   TArc *quarkCurved = new TArc(x1, y1, rad);
   quarkCurved->Draw();

   char result[7];
   strcpy(result, "#bar{");
   strncat(result, quarkName, 1);
   strcat(result, "}");

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, quarkName);
   t.DrawLatex(labelPositionX + 2 * rad, labelPositionY - 2 * rad, result);

   return quarkCurved;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Lepton
/// \param[in] x1 starting point in x direction
/// \param[in] y1 starting point in y direction
/// \param[in] x2 stopping point in x direction
/// \param[in] y2 stopping point in y direction
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
/// \param[in] whichLepton name of the Lepton (e, en, m, mn, t or tn -> Electron, Electron Neutrino ... )
/// \param[in] isMatter is the particle matter (true) or antimatter (false)
///
/// The TArrow is returned and the Label is drawn
TArrow *TFeynman::Lepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX,
                         Double_t labelPositionY, const char *whichLepton, bool isMatter)
{
   TArrow *e;

   e = new TArrow(x1, y1, x2, y2, 0.02, "->-");

   const char *usedLeptonName;
   if (isMatter == true) {
      if (whichLepton == std::string("e")) {
         usedLeptonName = "e^{-}";
      } else if (whichLepton == std::string("m")) {
         usedLeptonName = "#mu^{-}";
      } else if (whichLepton == std::string("t")) {
         usedLeptonName = "#tau^{-}";
      } else if (whichLepton == std::string("en")) {
         usedLeptonName = "#nu_{e}";
      } else if (whichLepton == std::string("mn")) {
         usedLeptonName = "#nu_{#mu}";
      } else if (whichLepton == std::string("tn")) {
         usedLeptonName = "#nu_{#tau}";
      }
   }

   if (isMatter == false) {
      if (whichLepton == std::string("e")) {
         usedLeptonName = "e^{+}";
      } else if (whichLepton == std::string("m")) {
         usedLeptonName = "#mu^{+}";
      } else if (whichLepton == std::string("t")) {
         usedLeptonName = "#tau^{+}";
      } else if (whichLepton == std::string("en")) {
         usedLeptonName = "\bar{#nu_{e}}";
      } else if (whichLepton == std::string("mn")) {
         usedLeptonName = "\bar{#nu_{#mu}}";
      } else if (whichLepton == std::string("tn")) {
         usedLeptonName = "\bar{#nu_{#tau}}";
      }
   }

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, usedLeptonName);

   return e;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Lepton Antilepton Pair
/// \param[in] x1 x coordinate of arc centre
/// \param[in] y1 y coordinate of arc centre
/// \param[in] rad radius of the arc
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
/// \param[in] whichLepton name of the quark (not the antiquark)
/// \param[in] whichAntiLepton name of the AntiLepton
///
/// The TArc is returned and the label is drawn
TArc *TFeynman::LeptonAntiLepton(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX,
                                 Double_t labelPositionY, const char *whichLepton, const char *whichAntiLepton)
{
   TArc *curvedLepton = new TArc(x1, y1, rad);

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, whichLepton);
   t.DrawLatex(labelPositionX + 2 * rad, labelPositionY - 2 * rad, whichAntiLepton);

   return curvedLepton;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Photon
/// \param[in] x1 starting point in x direction
/// \param[in] y1 starting point in y direction
/// \param[in] x2 stopping point in x direction
/// \param[in] y2 stopping point in y direction
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
///
/// The TCurlyLine is returned and the label is drawn

TCurlyLine *
TFeynman::Photon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY)
{
   TCurlyLine *gamma = new TCurlyLine(x1, y1, x2, y2);
   gamma->SetWavy();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, "#gamma");

   return gamma;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Photon that is later reabsorbed
/// \param[in] x1 x position of arc centre
/// \param[in] y1 y position of arc centre
/// \param[in] rad radius of the arc
/// \param[in] phimin minimum angle of arc (see TArc)
/// \param[in] phimax maximum angle of arc (see TArc)
/// \param[in] labelPositionX x position of label
/// \param[in] labelPositionY y position of label
///
/// The TCurlyArc is returned and the label is drawn
TCurlyArc *TFeynman::CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax,
                                  Double_t labelPositionX, Double_t labelPositionY)
{
   TCurlyArc *gammaCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
   gammaCurved->SetWavy();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, "#gamma");

   return gammaCurved;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Gluon
/// \param[in] x1 starting point in x direction
/// \param[in] y1 starting point in y direction
/// \param[in] x2 stopping point in x direction
/// \param[in] y2 stopping point in y direction
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
///
/// The TCurlyLine is returned and the label is drawn
TCurlyLine *
TFeynman::Gluon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY)
{
   TCurlyLine *gluon = new TCurlyLine(x1, y1, x2, y2);

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, "g");

   return gluon;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Gluon that is later reabsorbed
/// \param[in] x1 x position of arc centre
/// \param[in] y1 y position of arc centre
/// \param[in] rad radius of the arc
/// \param[in] phimin minimum angle of arc (see TArc)
/// \param[in] phimax maximum angle of arc (see TArc)
/// \param[in] labelPositionX x position of label
/// \param[in] labelPositionY y position of label
///
/// The TCurlyArc is returned and the label is drawn
TCurlyArc *TFeynman::CurvedGluon(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax,
                                 Double_t labelPositionX, Double_t labelPositionY)
{
   TCurlyArc *gCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
   gCurved->Draw();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, "g");

   return gCurved;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a weak force Boson
/// \param[in] x1 starting point in x direction
/// \param[in] y1 starting point in y direction
/// \param[in] x2 stopping point in x direction
/// \param[in] y2 stopping point in y direction
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
/// \param[in] whichWeakBoson name of the Weak Force Boson in Latex (Z_{0}, W^{+}, W^{-})
///
/// The TCurlyLine is returned and the label is drawn
TCurlyLine *TFeynman::WeakBoson(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX,
                                Double_t labelPositionY, const char *whichWeakBoson)
{
   TCurlyLine *weakBoson = new TCurlyLine(x1, y1, x2, y2);
   weakBoson->SetWavy();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, whichWeakBoson);

   return weakBoson;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Weak Force Boson that is later reabsorbed
/// \param[in] x1 x position of arc centre
/// \param[in] y1 y position of arc centre
/// \param[in] rad radius of the arc
/// \param[in] phimin minimum angle of arc (see TArc)
/// \param[in] phimax maximum angle of arc (see TArc)
/// \param[in] labelPositionX x position of label
/// \param[in] labelPositionY y position of label
/// \param[in] whichWeakBoson name of the Weak Force Boson in Latex (Z_{0}, W^{+}, W^{-})
///
/// The TCurlyArc is returned and the label is drawn
TCurlyArc *TFeynman::CurvedWeakBoson(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax,
                                     Double_t labelPositionX, Double_t labelPositionY, const char *whichWeakBoson)
{
   TCurlyArc *weakCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
   weakCurved->SetWavy();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, whichWeakBoson);

   return weakCurved;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Higgs Boson
/// \param[in] x1 starting point in x direction
/// \param[in] y1 starting point in y direction
/// \param[in] x2 stopping point in x direction
/// \param[in] y2 stopping point in y direction
/// \param[in] labelPositionX x coordinate of label
/// \param[in] labelPositionY y coordinate of label
///
/// The TCurlyLine is returned and the label is drawn
TCurlyLine *
TFeynman::Higgs(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY)
{
   TCurlyLine *higgs = new TCurlyLine(x1, y1, x2, y2);
   higgs->SetWavy();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, "H");

   return higgs;
}
////////////////////////////////////////////////////////////////////////////////
/// Draw a Higgs Boson that is later reabsorbed
/// \param[in] x1 x position of arc centre
/// \param[in] y1 y position of arc centre
/// \param[in] rad radius of the arc
/// \param[in] phimin minimum angle of arc (see TArc)
/// \param[in] phimax maximum angle of arc (see TArc)
/// \param[in] labelPositionX x position of label
/// \param[in] labelPositionY y position of label
///
/// The TCurlyArc is returned and the label is drawn
TCurlyArc *TFeynman::CurvedHiggs(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax,
                                 Double_t labelPositionX, Double_t labelPositionY)
{
   TCurlyArc *higgsCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
   higgsCurved->SetWavy();

   TLatex t;
   t.SetTextSize(0.1);
   t.DrawLatex(labelPositionX, labelPositionY, "H");

   return higgsCurved;
}
