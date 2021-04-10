#include "../inc/TFeynman.h"
#include <TCanvas.h>
#include <TLatex.h>
#include <TCurlyArc.h>
#include <TCurlyLine.h>
#include <TArc.h>
#include <TLine.h>

void TFeynman::Quark(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char * quarkName) {
    TArrow * q;
    
    q = new TArrow(x1, y1, x2, y2, 0.02, "->-");

    const char * usedQuarkName;
    
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
    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, usedQuarkName);

    q->Draw();
}

void TFeynman::Electron(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TArrow * e;
    
    e = new TArrow(x1, y1, x2, y2, 0.02, "->-");

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "e^{-}");

    e->Draw();
}

void TFeynman::Positron(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TArrow * eplus;

    eplus = new TArrow(x1, y1, x2, y2, 0.02, "-<-");

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "e^{-}");

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

void TFeynman::StraightWeakBoson(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY, const char *whichWeakBoson) {
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



