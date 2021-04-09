#include "TFeynman.h"
#include <TCanvas.h>
#include <TLatex.h>
#include <TCurlyArc.h>
#include <TCurlyLine.h>
#include <TArc.h>
#include <TLine.h>


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
    t.DrawLatex(labelPositionX, labelPositionY, "e^{+}");

    eplus->Draw();
}

void TFeynman::StraightPhoton(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyLine *gamma = new TCurlyLine(x1, y1, x2, y2);

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "#gamma");

    return *e;
}

void TFeynman::CurvedPhoton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY) {
    TCurlyArc *ginit = new TCurlyArc(x1, y1, rad, phimin, phimax);
    ginit->SetWavy();
    ginit->Draw();

    TLatex t;
    t.SetTextSize(0.1);
    t.DrawLatex(labelPositionX, labelPositionY, "#gamma");
}

