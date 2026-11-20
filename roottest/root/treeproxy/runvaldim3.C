#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TCanvas.h>

void runvaldim3()
{
   TFile file("forproxy.root");
   TTree *t; file.GetObject("t",t);

   // avoid extra message when default canvas is created
   // while such message can appear in random place relative to script ouput
   TCanvas c1("c0", "title", 500, 500);

   t->Process("val3dimSel.h+");
}