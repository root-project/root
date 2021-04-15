#include <cstdio>
#include <iostream>
#include "../inc/TFeynmanEntry.h"

ClassImp(TFeynman);


TFeynmanEntry::TFeynmanEntry(const char* particleName, Double_t x1, Double_t y1, Double_t x2, Double_t y2) {
    fX1 = x1;
    fY1 = y1;
    fX2 = x2;
    fY2 = y2;
    fParticle = particleName;
}
void TFeynmanEntry::Paint() {
  // Get all the needed values:
<<<<<<< HEAD
  cout << "Paint method of TFeynmanEntry called" << endl;
=======

>>>>>>> 7aad3df60e43a80be989dd65e792c4e4b670adc7
  Double_t x1 = GetX1();
  Double_t y1 = GetY1();
  Double_t x2 = GetX2();
  Double_t y2 = GetY2();
  const char* particleName = GetParticleName();

  if (particleName == std::string("fermion")) {
<<<<<<< HEAD
    cout << "Fermion" << endl;
    TArrow *fermion = new TArrow(x1, y1, x2, y2, 0.05, "->-");
    fermion->Paint();
  }
  else{
    cout << "Nope" << endl;
  }
=======
    TArrow *fermion = new TArrow(x1, y1, x2, y2, 0.05, "->-");
    fermion->Paint();
  }
>>>>>>> 7aad3df60e43a80be989dd65e792c4e4b670adc7
}
