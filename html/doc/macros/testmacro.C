#include "TCanvas.h"
#include "TH1F.h"

TObject* testmacro()
{

   TH1* h = new TH1F("h", "h", 100, 0., 1.);
   h->FillRandom("gaus",10000);
   TCanvas* c=new TCanvas("c","c");
   h->Draw();

   return c;
}
