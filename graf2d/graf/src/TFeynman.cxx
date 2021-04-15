#include <cstdio>
#include <iostream>
#ifndef ROOT_TFeynman
#include "../inc/TFeynman.h"

#include "TMath.h"
#include "TCurlyLine.h"

#include "TObject.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TVirtualPad.h"
#include "TROOT.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TH1.h"
#include "THStack.h"
#include "TArrow.h"

ClassImp(TFeynman);

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



TFeynman::TFeynman(Double_t canvasWidth, Double_t canvasHeight){
				TCanvas *c1 = new TCanvas("c1", "c1", 10,10, canvasWidth, canvasHeight);
   			c1->Range(0, 0, 140, 60);
				gStyle->SetLineWidth(2);
        fPrimitives = new TList();
		}

TFeynmanEntry* TFeynman::AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2) {
   TFeynmanEntry *newEntry = new TFeynmanEntry(particleName, x1, y1, x2, y2);
   //fPrimitives->Add(newEntry);
   return newEntry;
}

void TFeynman::Draw() {
	AppendPad();
}
void TFeynman::Paint() {

	TIter next(fPrimitives);
	TFeynmanEntry *entry;
	Int_t iColumn = 0;
	const char* particle = entry->GetParticleName();
	while (( entry = (TFeynmanEntry*)next() )) {
		entry->Paint();
	}

}
#endif
