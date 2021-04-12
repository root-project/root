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
  f->Lepton(10, 10, 30, 30, 7, 6, "e", true);
  f->Lepton(10, 50, 30, 30, 5, 55, "e", false);
  f->CurvedPhoton(30, 30, 12.5*sqrt(2), 135, 225, 7, 30);
  f->Photon(30, 30, 55, 30, 42.5, 37.7);
  f->QuarkAntiQuark(70, 30, 15, 55, 45, "q");
  f->Gluon(70, 45, 70, 15, 77.5, 30);
  f->WeakBoson(85, 30, 110, 30, 100, 37.5, "Z^{0}");
  f->Quark(110, 30, 130, 50, 135, 55, "q", true);
  f->Quark(110, 30, 130, 10, 135, 6, "q", false);
  f->CurvedGluon(110, 30, 12.5*sqrt(2), 315, 45, 135, 30);
~~~

No drawing is needed, as this is all done in the method itself. 
One can simply use the methods like Legos.

*/

#include "../inc/TFeynman.h"
#include <TCanvas.h>
#include <TLatex.h>
#include <TCurlyArc.h>
#include <TCurlyLine.h>
#include <TArc.h>
#include <TLine.h>
#include <string.h>

void TFeynman::Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName, bool isMatter) {
    TArrow * q;
    
    q = new TArrow(x1, y1, x2, y2, 0.02, "->-");

    const char * usedQuarkName;

    if (isMatter == true) {
        if (quarkName == std::string("q")) {
            usedQuarkName = "q";
        }
        else if (quarkName == std::string("u")) {
            usedQuarkName = "u";
        }
        else if (quarkName == std::string("d")) {
            usedQuarkName = "d";
        }
        else if (quarkName == std::string("c")) {
            usedQuarkName = "c";
        }
        else if (quarkName == std::string("s")) {
            usedQuarkName = "s";
        }
        else if (quarkName == std::string("t")) {
            usedQuarkName = "t";
        }
        else if (quarkName == std::string("b")) {
            usedQuarkName = "b";
        }
        else{
            usedQuarkName = "q";
        }
    }

    if (isMatter == false) {
        if (quarkName == std::string("q")) {
        usedQuarkName = "#bar{q}";
        }
        else if (quarkName == std::string("u")) {
        usedQuarkName = "#bar{u}";
        }
        else if (quarkName == std::string("d")) {
        usedQuarkName = "#bar{d}";
        }
        else if (quarkName == std::string("c")) {
        usedQuarkName = "#bar{c}";
        }
        else if (quarkName == std::string("s")) {
        usedQuarkName = "#bar{s}";
        }
        else if (quarkName == std::string("t")) {
        usedQuarkName = "#bar{t}";
        }
        else if (quarkName == std::string("b")) {
        usedQuarkName = "#bar{b}";
        }
        else{
        usedQuarkName = "#bar{b}";
        }
    }
    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, usedQuarkName);

    q->Draw();
}

void TFeynman::QuarkAntiQuark(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName) {
    TArc *quarkCurved = new TArc(x1, y1, rad);
    quarkCurved->Draw();

    char result[7];
    strcpy(result, "#bar{");
    strncat(result, quarkName, 1);
    strcat(result, "}");

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, quarkName);
    t.DrawLatex(labelPositionX + 2*rad, labelPositionY - 2*rad, result);
}

void TFeynman::Lepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, bool isMatter) {
    TArrow * e;
    
    e = new TArrow(x1, y1, x2, y2, 0.02, "->-");

    const char * usedLeptonName;
    if (isMatter == true){
        if (whichLepton == std::string("e")) {
            usedLeptonName = "e^{-}";
        }
        else if (whichLepton == std::string("m")) {
            usedLeptonName = "#mu^{-}";
        }
        else if (whichLepton == std::string("t")) {
            usedLeptonName = "#tau^{-}";
        }
        else if (whichLepton == std::string("en")) {
            usedLeptonName = "#nu_{e}";
        }
        else if (whichLepton == std::string("mn")) {
            usedLeptonName = "#nu_{#mu}";
        }
        else if (whichLepton == std::string("tn")) {
            usedLeptonName = "#nu_{#tau}";
        }
    }

    if (isMatter == false) {
        if (whichLepton == std::string("e")) {
        usedLeptonName = "e^{+}";
        }
        else if (whichLepton == std::string("m")) {
        usedLeptonName = "#mu^{+}";
        }
        else if (whichLepton == std::string("t")) {
        usedLeptonName = "#tau^{+}";
        } 
        else if (whichLepton == std::string("en")) {
            usedLeptonName = "\bar{#nu_{e}}";
        }
        else if (whichLepton == std::string("mn")) {
            usedLeptonName = "\bar{#nu_{#mu}}";
        }
        else if (whichLepton == std::string("tn")) {
            usedLeptonName = "\bar{#nu_{#tau}}";
        }
    }

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, usedLeptonName);

    e->Draw();
}

void TFeynman::LeptonAntiLepton(Double_t x1, Double_t y1, Double_t rad, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, const char * whichAntiLepton) {
    TArc *curvedLepton = new TArc(x1, y1, rad);
    curvedLepton->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, whichLepton);
    t.DrawLatex(labelPositionX + 2*rad, labelPositionY - 2*rad, whichAntiLepton);
}

void TFeynman::Photon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyLine *gamma = new TCurlyLine(x1, y1, x2, y2);
    gamma->SetWavy();
    gamma->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "#gamma");

}

void TFeynman::CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyArc *gammaCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
    gammaCurved->SetWavy();
    gammaCurved->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "#gamma");
}

void TFeynman::Gluon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyLine *gluon = new TCurlyLine(x1, y1, x2, y2);
    gluon->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "g");
}

void TFeynman::CurvedGluon(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyArc *gCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
    gCurved->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "g");
}

void TFeynman::WeakBoson(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char *whichWeakBoson) {
    TCurlyLine *weakBoson = new TCurlyLine(x1, y1, x2, y2);
    weakBoson->SetWavy();
    weakBoson->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, whichWeakBoson);
}

void TFeynman::CurvedWeakBoson(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY, const char* whichWeakBoson) {
    TCurlyArc *weakCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
    weakCurved->SetWavy();
    weakCurved->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, whichWeakBoson);
}

void TFeynman::Higgs(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyLine *higgs = new TCurlyLine(x1, y1, x2, y2);
    higgs->SetWavy();
    higgs->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "H");
}

void TFeynman::CurvedHiggs(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyArc *higgsCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
    higgsCurved->SetWavy();
    higgsCurved->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "H");
}



