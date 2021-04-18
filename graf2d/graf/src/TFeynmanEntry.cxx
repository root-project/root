#include <cstdio>
#include <iostream>
#include "../inc/TFeynmanEntry.h"

ClassImp(TFeynmanEntry);

/** \class TFeynmanEntry
\ingroup BasicGraphics

  Storage Class for TFeynman
*/


///////////////////////////////////////////////////////////////////////////////
/// Constructor

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
     boson->Draw();

     TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
     t->Paint();
   }
   else if (particleName == std::string("gluon")) {
     TCurlyLine *gluon = new TCurlyLine(x1, y1, x2, y2);
     gluon->Draw();

     TLatex *t = new TLatex(fLabelX, fLabelY, fLabel);
     t->Paint();
   }
   else{
     Error("TFeynmanEntry::Paint()", "Invalid Particle!");
   }

}
