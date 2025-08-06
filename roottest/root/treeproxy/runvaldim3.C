#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>

void runvaldim3()
{
   TFile file("forproxy.root");
   TTree *t; file.GetObject("t",t);
   t->Process("val3dimSel.h+");
}