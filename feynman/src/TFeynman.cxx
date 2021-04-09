#include "../inc/TFeynman.h"
#include <TCanvas.h>
#include <TLatex.h>
#include <TCurlyArc.h>
#include <TCurlyLine.h>
#include <TArc.h>
#include <TLine.h>

void TFeynman::Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, Double_t quarkName) {
    TArrow * q;
    
    q = new TArrow(x1, y1, x2, y2, 0.02, "->-");
    
    const char *quark;

    if (quarkName == 0) {
        quark = "q";
    }
    else if (quarkName == 1) {
        quark = "u";
    }
    else if (quarkName == 2) {
        quark = "d";
    }
    else if (quarkName == 3) {
        quark = "s";
    }
    else if (quarkName == 4) {
        quark = "s";
    }
    else if (quarkName == 5) {
        quark = "t";
    }
    else if (quarkName == 6) {
        quark = "b";
    }

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, quark);

    q->Draw();
}

void TFeynman::Lepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, Double_t leptonName) {
    TArrow * e;
    
    e = new TArrow(x1, y1, x2, y2, 0.02, "->-");
    
    const char *lepton;

    if (leptonName == 1) {
        lepton = "e^{-}";
    }
    else if (leptonName == 2) {
        lepton = "#mu^{-}";
    }
    else if (leptonName == 3) {
        lepton = "#tau^{-}";
    }

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, lepton);

    e->Draw();
}

void TFeynman::AntiLepton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, Double_t leptonName) {
    TArrow * eplus;

    eplus = new TArrow(x1, y1, x2, y2, 0.02, "-<-");

    const char *lepton;

    if (leptonName == 1) {
        lepton = "e^{+}";
    }
    else if (leptonName == 2) {
        lepton = "#mu^{+}";
    }
    else if (leptonName == 3) {
        lepton = "#tau^{+}";
    }

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, lepton);

    eplus->Draw();
}

void TFeynman::StraightPhoton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
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

void TFeynman::StraightGluon(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyLine *gluon = new TCurlyLine(x1, y1, x2, y1);
    gluon->Draw();

    TLatex t;
    t.DrawLatex(labelPositionX, labelPositionY, "g");
}

void TFeynman::CurvedGluon(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyArc *gCurved = new TCurlyArc(x1, y1, rad, phimin, phimax);
    gCurved->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "g");
}



