/// \file
/// \ingroup tutorial_tree
/// \notebook
/// TSpider example.
///
/// \macro_code
///
/// \author Bastien Dallapiazza

#include "TFile.h"
#include "TCanvas.h"
#include "TNtuple.h"
#include "TSpider.h"

void spider() {
   TCanvas *c1 = new TCanvas("c1","TSpider example",200,10,700,700);
   TFile *f = new TFile("hsimple.root");
   if (!f || f->IsZombie()) {
      printf("Please run <ROOT location>/tutorials/hsimple.C before.");
      return;
   }
   TNtuple* ntuple = (TNtuple*)f->Get("ntuple");
   TString varexp = "px:py:pz:random:sin(px):log(px/py):log(pz)";
   TString selection = "px>0 && py>0 && pz>0";
   TString options = "average";
   TSpider *spider = new TSpider(ntuple,varexp.Data(),selection.Data(),options.Data());
   spider->Draw();
   c1->ToggleEditor();
   c1->Selected(c1,spider,1);
}
