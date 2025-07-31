#define Tree_cxx

#include "generated_selectors/Tree.h"
#include <TH2.h>
#include <TStyle.h>

void Tree::Begin(TTree * /*tree*/)
{
   TString option = GetOption();
}

void Tree::SlaveBegin(TTree * /*tree*/)
{
   TString option = GetOption();
}

Bool_t Tree::Process(Long64_t entry)
{
   fReader.SetEntry(entry);

   fprintf(stderr, "%.1f %.1f %.1f %.1lf %d\n", *px, *py, *py, *random, *ev);

   return kTRUE;
}

void Tree::SlaveTerminate() { }

void Tree::Terminate() { }
