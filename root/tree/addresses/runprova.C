#include "TROOT.h"
#include "TMinuit.h"
#include <stdio.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>

int runprova()
{

   TChain *chain_null = new TChain("Eventi");
   chain_null->Add("prova1.root");
   chain_null->Add("prova2.root");
   chain_null->Draw("br.n_run");
   chain_null->Merge("merge.root");
   TFile *f = new TFile("merge.root");
   TTree *t; f->GetObject("Eventi",t);
   t->Scan("br.n_ev");

   return 0;
}
