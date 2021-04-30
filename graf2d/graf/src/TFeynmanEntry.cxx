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
#include "TFeynmanEntry.h"

ClassImp(TFeynmanEntry);

/** \class TFeynmanEntry
\ingroup BasicGraphics

  Storage Class for TFeynman
*/


///////////////////////////////////////////////////////////////////////////////
/// Constructors

///////////////////////////////////////////////////////////////////////////////
/// Normal particle constuctor
/// \param[in] particleName name of the particle (boson, fermion, gluon, anti-fermion)
/// \param[in] x1
/// \param[in] y1 starting coordinates of the particle
/// \param[in] x2
/// \param[in] y2 stopping coordinates of the particle
/// \param[in] labelX
/// \param[in] labelY coordinates of label
/// \param[in] label to be displayed in Latex form

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

///////////////////////////////////////////////////////////////////////////////
/// Particle pair constuctor
/// \param[in] particleLabel label to be displayed (kust the particle, not the antiparticle)
/// \param[in] x
/// \param[in] y coordinates of the centre of the pair
/// \param[in] radius radius of the arc

TFeynmanEntry::TFeynmanEntry(const char *particleLabel, Double_t x, Double_t y, Double_t radius) {
  fX1 = x;
  fY1 = y;
  fRadius = radius;
  fParticle = "pair";
  fLabel = particleLabel;

}

///////////////////////////////////////////////////////////////////////////////
/// Curved particle constuctor
/// \param[in] particleLabel label to be displayed
/// \param[in] x
/// \param[in] y coordinates of the centre of the curve
/// \param[in] radius radius of the arc
/// \param[in] phimin minimum angle (see TArc)
/// \param[in] phimax maximum angle (see TArc)

TFeynmanEntry::TFeynmanEntry(const char *particleLabel, Double_t x, Double_t y, Double_t radius, Double_t phimin, Double_t phimax, bool wavy) {
  fParticle = "curved";
  fX1 = x;
  fY1 = y;
  fRadius = radius;
  fLabel = particleLabel;
  fPhimin = phimin;
  fPhimax = phimax;
  fWavy = wavy;
}


///////////////////////////////////////////////////////////////////////////////
/// Paint Method

void TFeynmanEntry::Paint( Option_t* )
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
      delete fermion;
   }
   else if (particleName == std::string("anti-fermion")) {
      TArrow *fermion = new TArrow(x1, y1, x2, y2, 0.03, "-<-");
      fermion->Paint();

      TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
      t->Paint();

      delete fermion;
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
   else if (fParticle == std::string("pair")) {
     TArc *particlePair = new TArc(fX1, fY1, fRadius);
     particlePair->Paint();

     const char* antiparticle = "#bar{" + fLabel + "}";
     TLatex *t = new TLatex(fX1 - 0.85 * fRadius, fY1 + 0.85*fRadius, fLabel);
     TLatex *u = new TLatex(fX1 + 0.85 * fRadius, fY1 - 0.85*fRadius, antiparticle);

     t->Paint();
     u->Paint();
   }
   else if (fParticle == std::string("curved")) {
      TCurlyArc *curved = new TCurlyArc(fX1, fY1, fRadius, fPhimin, fPhimax);
      if (fWavy == true) {
        curved->SetWavy();
      }
      curved->Paint();

      TLatex *t = new TLatex(fX1, fY1, fLabel);
      t->Paint();
   }
   else{
     Error("TFeynmanEntry::Paint()", "Invalid Particle!");
   }

}
