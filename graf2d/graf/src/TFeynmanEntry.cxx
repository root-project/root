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
#include "../inc/TFeynmanEntry.h"

ClassImp(TFeynmanEntry);

/** \class TFeynmanEntry
\ingroup BasicGraphics

  Storage Class for TFeynman
*/


///////////////////////////////////////////////////////////////////////////////
/// Constructors

TFeynmanEntry::TFeynmanEntry(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t labelX, Double_t labelY, const char* label) {
   fX1 = x1;
   fY1 = y1;
   fX2 = x2;
   fY2 = y2;
   fLabelX = labelX;
   fLabelY = labelY;
   fParticle = particleName;
   fLabel = label;
}

TFeynmanEntry::TFeynmanEntry(const char *particleLabel, Double_t x, Double_t y, Double_t radius) {
  fX1 = x;
  fY1 = y;
  fRadius = radius;
  fParticle = "pair";


}


///////////////////////////////////////////////////////////////////////////////
/// Paint Method

void TFeynmanEntry::Paint( Option_t* option )
{
   // Get all the needed values:
   Double_t x1 = GetX1();
   Double_t y1 = GetY1();
   Double_t x2 = GetX2();
   Double_t y2 = GetY2();
   const char* particleName = GetParticleName();

   if (particleName == std::string("fermion")) {
      TArrow *fermion = new TArrow(x1, y1, x2, y2, 0.03, "->-");
      fermion->Paint();

      TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
      t->Paint();
   }
   else if (particleName == std::string("anti-fermion")) {
      TArrow *fermion = new TArrow(x1, y1, x2, y2, 0.03, "-<-");
      fermion->Paint();

      TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
      t->Paint();
   }
   else if (particleName == std::string("boson")) {
     TCurlyLine *boson = new TCurlyLine(x1, y1, x2, y2);
     boson->SetWavy();
     boson->Paint();

     TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
     t->Paint();
   }
   else if (particleName == std::string("gluon")) {
     TCurlyLine *gluon = new TCurlyLine(x1, y1, x2, y2);
     gluon->Paint();

     TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
     t->Paint();
   }
   else if (particleName == std::string("pair")) {
     TArc *particlePair = new TArc(fX1, fY1, fRadius);
     particlePair->Paint();
   }
   else{
     Error("TFeynmanEntry::Paint()", "Invalid Particle!");
   }

}
