#include "./TFeynman.cxx"
#include "TFeynmanEntry.cxx"
void feynman()
{
  TFeynman *f = new TFeynman(600, 300);


  f->AddItem("fermion", 10, 10, 30, 30);

  f->Draw();

}
