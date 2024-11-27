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

void tree140_spider()
{
   auto c1 = new TCanvas("c1", "TSpider example", 200, 10, 700, 700);
   auto f = TFile::Open("hsimple.root");
   if (!f || f->IsZombie()) {
      printf("Please run <ROOT location>/tutorials/hsimple.C before.");
      return;
   }
   auto ntuple = f->Get<TNtuple>("ntuple");
   TString varexp = "px:py:pz:random:sin(px):log(px/py):log(pz)";
   TString selection = "px>0 && py>0 && pz>0";
   TString options = "average";
   auto spider = new TSpider(ntuple, varexp.Data(), selection.Data(), options.Data());
   spider->Draw();
   c1->ToggleEditor();
   c1->Selected(c1, spider, 1);
}
