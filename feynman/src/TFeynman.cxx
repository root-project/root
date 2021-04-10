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
    strcat(result, quarkName);
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

void CurvedLepton(Double_t x1, Double_t y1, Double_t rad, Double_t phimin, Double_t phimax, Double_t labelPositionX, Double_t labelPositionY, const char * whichLepton, bool isMatter) {
    TArc *curvedLepton = new TArc(x1, y1, rad, phimin, phimax);
    curvedLepton->Draw();

    const char * usedLeptonName;

    if (isMatter == true) {
        if (whichLepton == std::string("e")) {
            usedLeptonName = "e_{-}";
        }
        else if (whichLepton == std::string("m")) {
            usedLeptonName = "#mu_{-}";
        }
        else if (whichLepton == std::string("t")) {
            usedLeptonName = "#tau_{-}";
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
        usedLeptonName = "e_{+}";
        }
        else if (whichLepton == std::string("m")) {
        usedLeptonName = "#mu_{+}";
        }
        else if (whichLepton == std::string("t")) {
        usedLeptonName = "#tau_{+}";
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



