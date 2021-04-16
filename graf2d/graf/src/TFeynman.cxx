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
#include "TArrow.h"
#include "TList.h"


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



TFeynman::TFeynman() : TAttLine(){
				gStyle->SetLineWidth(2);
        fPrimitives = new TList;
		}

TFeynmanEntry *TFeynman::AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2) {
   TFeynmanEntry *newEntry = new TFeynmanEntry(particleName, x1, y1, x2, y2);
	 if ( !fPrimitives ) fPrimitives = new TList;
	 std::cout << "Added " << newEntry->GetParticleName() << " to the Feynman Diagram" << std::endl;
   fPrimitives->Add(newEntry);

   return newEntry;
}

void TFeynman::Draw(){
	std::cout << "Draw Method called. Grab your pencils." << std::endl;
	AppendPad();
}
void TFeynman::Paint() {
	std::cout << "Paint Method called. Grab your paintbrush" << std::endl;
	TIter next(fPrimitives);
	TFeynmanEntry *entry;
	while (( entry = (TFeynmanEntry*)next() )) {
		entry->Paint();
	}

}
#endif
