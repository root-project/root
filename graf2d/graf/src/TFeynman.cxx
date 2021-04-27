// @(#)root/graf:$Id$
// Author: Advait Dhingra and Oliver Couet 18/04/21

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
   else {
      Error("TFeynman::TFeynman()", "Error. You need to create a canvas or gPad first.");
      return;
   };
   fPrimitives = new TList;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a particle to TFeynman
/// \param[in] particleName name of the particle (boson, fermion, gluon, anti-fermion)
/// \param[in] x1
/// \param[in] y1 starting coordinates of the particle
/// \param[in] x2
/// \param[in] y2 stopping coordinates of the particle
/// \param[in] labelX
/// \param[in] labelY coordinates of label
/// \param[in] label to be displayed in Latex form

TFeynmanEntry *TFeynman::AddItem(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelX, Double_t labelY, const char* label)
{
   TFeynmanEntry *newEntry = new TFeynmanEntry(particleName, x1, y1, x2, y2, labelX, labelY, label);
   if ( !fPrimitives ) fPrimitives = new TList;
   fPrimitives->Add(newEntry);

   return newEntry;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a particle pair to TFeynman
/// \param[in] particleLabel label to be displayed (kust the particle, not the antiparticle)
/// \param[in] x
/// \param[in] y coordinates of the centre of the pair
/// \param[in] radius radius of the arc
TFeynmanEntry *TFeynman::AddPair(const char *particleLabel, Double_t x, Double_t y, Double_t radius) {
  TFeynmanEntry *newPairEntry = new TFeynmanEntry(particleLabel, x, y, radius);
  if (!fPrimitives) fPrimitives = new TList;
  fPrimitives->Add(newPairEntry);

  return newPairEntry;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a curved particle to TFeynman
/// \param[in] particleLabel label to be displayed
/// \param[in] x
/// \param[in] y coordinates of the centre of the curve
/// \param[in] radius radius of the arc
/// \param[in] phimin minimum angle (see TArc)
/// \param[in] phimax maximum angle (see TArc)
TFeynmanEntry *TFeynman::AddCurved(const char *particleLabel, Double_t x, Double_t y, Double_t radius, Double_t phimin, Double_t phimax, bool wavy) {
  TFeynmanEntry *newCurvedEntry = new TFeynmanEntry(particleLabel, x, y, radius, phimin, phimax, wavy);
  if (!fPrimitives) fPrimitives = new TList;
  fPrimitives->Add(newCurvedEntry);

  return newCurvedEntry;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this Feynman's diagram with its current attributes.
/// \param[in] option drawing options
void TFeynman::Draw( Option_t *option )
{
   AppendPad(option);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint Method

void TFeynman::Paint( Option_t*)
{
   TIter next(fPrimitives);
   TFeynmanEntry *entry;
   while (( entry = (TFeynmanEntry*)next() )) {
      entry->Paint();
   }
}
#endif
