{
// create template class for Selector to run on a tree 
//////////////////////////////////////////////////////
//
// open root file containing the Tree
  TFile *f = TFile::Open("conductivity_experiment.root"); 
// create TTree object from it
  TTree *t = (TTree *) f->Get("cond_data");
// this generates the files MySelector.h and MySelector.C 
  t->MakeSelector("MySelector");
  }
