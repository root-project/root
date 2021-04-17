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
// Author: Advait Dhingra and Oliver Couet   12/04/2021

/** \class TFeynman
    \ingroup BasicGraphics
TFeynman is a class that makes it easier to make
good-looking Feynman Diagrams using ROOT components
like TArc and TArrow.
### Decleration / Access to the components
To initialize TFeynman:
~~~
  TFeynman *f = new TFeynman();
~~~
Here is an example of how TFeynman can be used:
~~~
TCanvas *c1 = new TCanvas();
TFeynman *f = new TFeynman();

// proton decay (beta minus)

f->AddItem("fermion", 10, 10, 30, 30, 5, 6, "d");
f->AddItem("fermion", 30, 30, 10, 50, 5, 50, "d");
f->AddItem("fermion", 15, 10, 35, 30, 10, 6, "u");
f->AddItem("fermion", 35, 30, 15, 50, 12, 50, "u");
f->AddItem("fermion", 20, 10, 40, 30, 15, 6, "u");
f->AddItem("fermion", 40, 30, 20, 50, 17, 50, "d");
f->AddItem("boson", 40, 30, 70, 30, 55, 35, "W^{+}");
f->AddItem("anti-fermion", 70, 30, 90, 50, 95, 55, "e^{+}");
f->AddItem("fermion", 70, 30, 90, 10, 85, 5, "#bar{#nu}");
f->Draw();
~~~

*/


////////////////////////////////////////////////////////////////////////////////
/// Constructor

TFeynman::TFeynman() : TAttLine(){
   gStyle->SetLineWidth(2);
   if (gPad)
      gPad->Range(0, 0, 140, 60);
   fPrimitives = new TList;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a particle to TFeynman

TFeynmanEntry *TFeynman::AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelX, Double_t labelY, const char* label)
{
   TFeynmanEntry *newEntry = new TFeynmanEntry(particleName, x1, y1, x2, y2, labelX, labelY, label);
   if ( !fPrimitives ) fPrimitives = new TList;
   fPrimitives->Add(newEntry);

   return newEntry;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw this Feynman's diagram with its current attributes.

void TFeynman::Draw( Option_t *option )
{
   AppendPad(option);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint Method 

void TFeynman::Paint( Option_t* option )
{
   TIter next(fPrimitives);
   TFeynmanEntry *entry;
   while (( entry = (TFeynmanEntry*)next() )) {
      entry->Paint();
   }
}
#endif
