#ifndef R__templateName
#define R__templateName
// Make proxy does not clean duplicate header file

#include <TTree.h>
#include <TFile.h>

template <class T>
struct B { int i; };

struct C : B<void> { int j; };


void plot_my_i()
{
  C c, *cp(&c);
  
  TTree *t = new TTree("mytree", "Icecube", 10000);

  t->Branch("my_c", "C", &cp);
  t->Branch("my_c2", "C", &cp);
  t->Branch("plot_my_i.", "C", &cp);
  t->Branch("against_my_i.", "C", &cp);

  for (int i=0; i<100; i++)
    {
      c.i=i*13;
      t->Fill();
    }
  t->Print();
  t->Draw("forproxy.C+","","goff");
  t->Draw("plot_my_i.B<void>.i:against_my_i.B<void>.i");
  //t->StartViewer();
}

struct C2 { int i; };
void makeclass()
{
  C2 c, *cp(&c);
  TTree *t = new TTree("mytree", "foo", 10000);
  t->Branch("my_c", "C2", &cp);
  t->Branch("other_c.", "C2", &cp);
  t->Fill();
  t->Print();
  t->MakeClass("cmakeclass");
  t->MakeProxy("cmakeproxy","forproxy.C");
  t->Draw("forproxy.C+","","goff");
}

void templateName() { plot_my_i(); makeclass(); };

#endif
