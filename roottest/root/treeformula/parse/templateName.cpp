#ifndef R__templateName
#define R__templateName
// Make proxy does not clean duplicate header file

#include <TTree.h>
#include <TFile.h>

#include "TSystem.h"

template <class T>
struct B { int i; };

class C : public B<void> { public: int j; };


void plot_my_i()
{
  C c, *cp(&c);
  C c2, *cp2(&c2);

  TFile *f = TFile::Open("mytree.root", "recreate");
  TTree *t = new TTree("mytree", "Icecube", 10000);

  t->Branch("my_c", "C", &cp);
  t->Branch("my_c2", "C", &cp2);
  t->Branch("plot_my_i.", "C", &cp);
  t->Branch("against_my_i.", "C", &cp2);

  for (int i=0; i<100; i++)
    {
      c.i=i*13;
      c2.i=i*26;
      t->Fill();
    }
  t->Print();
  t->Draw("forproxy.C+","","goff");
  t->Scan("plot_my_i.B<void>.i:against_my_i.B<void>.i","","",10);
  t->Scan("plot_my_i.i:against_my_i.i","","",10);
  t->ResetBranchAddresses();
  //t->StartViewer();

  delete t;
  delete f;
}

class C2 { public: int i; };
void makeclass()
{
  C2 c, *cp(&c);
  TFile *f = TFile::Open("mytreemk.root", "recreate");
  TTree *t = new TTree("mytreemk", "foo", 10000);
  t->Branch("my_c", "C2", &cp);
  t->Branch("other_c.", "C2", &cp);
  t->Fill();
  t->Print();
  t->MakeClass("cmakeclass");
  t->MakeProxy("cmakeproxy","forproxy.C");
  gSystem->CopyFile("generatedSel.h","generatedSel.old");
  t->Draw("forproxy.C+","","goff");
  gSystem->Exec("diff generatedSel.old generatedSel.h");
  t->ResetBranchAddresses();
  delete t;
  delete f;
}

void templateName() { plot_my_i(); makeclass(); }

#endif
