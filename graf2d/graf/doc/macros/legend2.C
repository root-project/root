#include "TCanvas.h"
#include "TH1F.h"
#include "TLegend.h"

TCanvas* legend2()
{
   TCanvas *c2 = new TCanvas("c2","c2",500,300);

   TLegend* leg = new TLegend(0.2, 0.2, .8, .8);
   TH1* h = new TH1F("", "", 1, 0, 1);

   leg->AddEntry(h, "Histogram \"h\"", "l");
   leg->AddEntry((TObject*)0, "", "");
   leg->AddEntry((TObject*)0, "Some text", "");
   leg->AddEntry((TObject*)0, "", "");
   leg->AddEntry(h, "Histogram \"h\" again", "l");

   leg->Draw();
   return c2;
}

