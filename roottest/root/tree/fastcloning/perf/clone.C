#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include "Riostream.h"

void clone(char *filename, char *tag) 
{
  TString opt(tag);
  opt.Prepend("fast,");

  TString newfile = gSystem->BaseName(filename);

  gSystem->Load("libEvent.so");
  TFile *f0 = TFile::Open(filename);
  TTree *from = (TTree*) f0->Get("T");
  TFile *f1 = new TFile("clone.root","RECREATE");
  cerr << "The option are: " << opt << endl;;
  from->CloneTree(-1,opt);
  f1->Write();
  delete f0; delete f1;
  gSystem->Exec(Form("mv clone.root %s", newfile.Data()));
}

