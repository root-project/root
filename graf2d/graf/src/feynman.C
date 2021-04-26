#include "TFeynman.cxx"
#include "TFeynmanEntry.cxx"

void feynman()
{
  TCanvas *c1 = new TCanvas();
  TFeynman *f = new TFeynman();

  // proton decay (beta minus)
  f->AddPair("q", 50, 30, 6);
  f->AddItem("fermion", 10, 10, 30, 30, 5, 6, "d");
  f->AddItem("fermion", 30, 30, 10, 50, 5, 50, "d");
  f->AddItem("fermion", 15, 10, 35, 30, 10, 6, "u");
  f->AddItem("fermion", 35, 30, 15, 50, 12, 50, "u");
  f->AddItem("fermion", 20, 10, 40, 30, 15, 6, "u");
  f->AddItem("fermion", 40, 30, 20, 50, 17, 50, "d");
  f->AddItem("boson", 40, 30, 70, 30, 55, 35, "W^{+}");
  f->AddItem("anti-fermion", 70, 30, 90, 50, 95, 55, "e^{+}");
  f->AddItem("fermion", 70, 30, 90, 10, 85, 5, "#bar{#nu}");
  f->AddCurved("e^{-}", 50, 40, 12, 90, 275);
  f->Draw();
}
