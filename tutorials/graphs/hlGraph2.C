/// \file
/// \ingroup tutorial_graphs
///
/// This tutorial demonstrates how to use the highlight mode on graph.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

TNtuple *ntuple = nullptr;

void HighlightBinId(TVirtualPad *pad, TObject *obj, Int_t ihp, Int_t y)
{
   auto Canvas2 = (TCanvas *)gROOT->GetListOfCanvases()->FindObject("Canvas2");
   if (!Canvas2) return;
   auto histo = (TH1F *)Canvas2->FindObject("histo");
   if (!histo) return;

   Double_t px = ntuple->GetV1()[ihp];
   Double_t py = ntuple->GetV2()[ihp];
   Double_t pz = ntuple->GetV3()[ihp];
   Double_t i  = ntuple->GetV4()[ihp];
   Double_t p  = TMath::Sqrt(px*px + py*py + pz*pz);
   Int_t hbin = histo->FindBin(p);

   Bool_t redraw = kFALSE;
   auto bh = (TBox *)Canvas2->FindObject("TBox");
   if (!bh) {
      bh = new TBox();
      bh->SetFillColor(kBlack);
      bh->SetFillStyle(3001);
      bh->SetBit(kCannotPick);
      bh->SetBit(kCanDelete);
      redraw = kTRUE;
   }
   bh->SetX1(histo->GetBinLowEdge(hbin));
   bh->SetY1(histo->GetMinimum());
   bh->SetX2(histo->GetBinWidth(hbin) + histo->GetBinLowEdge(hbin));
   bh->SetY2(histo->GetBinContent(hbin));

   auto th = (TText *)Canvas2->FindObject("TText");
   if (!th) {
      th = new TText();
      th->SetName("TText");
      th->SetTextColor(bh->GetFillColor());
      th->SetBit(kCanDelete);
      redraw = kTRUE;
   }
   th->SetText(histo->GetXaxis()->GetXmax()*0.75, histo->GetMaximum()*0.5,
               TString::Format("id = %d", (Int_t)i));

   if (ihp == -1) { // after highlight disabled
      delete bh;
      delete th;
   }
   Canvas2->Modified();
   Canvas2->Update();
   if (!redraw) return;

   auto savepad = gPad;
   Canvas2->cd();
   bh->Draw();
   th->Draw();
   Canvas2->Update();
   savepad->cd();
}

void hlGraph2()
{
   auto dir = gROOT->GetTutorialDir();
   dir.Append("/hsimple.C");
   dir.ReplaceAll("/./","/");
   if (!gInterpreter->IsLoaded(dir.Data())) gInterpreter->LoadMacro(dir.Data());
   auto file = (TFile*)gROOT->ProcessLineFast("hsimple(1)");
   if (!file) return;

   file->GetObject("ntuple", ntuple);
   if (!ntuple) return;

   TCanvas *Canvas1 = new TCanvas("Canvas1", "Canvas1", 0, 0, 500, 500);
   Canvas1->HighlightConnect("HighlightBinId(TVirtualPad*,TObject*,Int_t,Int_t)");

   const char *cut = "pz > 3.0";
   ntuple->Draw("px:py", cut);
   TGraph *graph = (TGraph *)gPad->FindObject("Graph");

   auto info = new TText(0.0, 4.5, "please move the mouse over the graph");
   info->SetTextAlign(22);
   info->SetTextSize(0.03);
   info->SetTextColor(kRed+1);
   info->SetBit(kCannotPick);
   info->Draw();

   graph->SetHighlight();

   auto Canvas2 = new TCanvas("Canvas2", "Canvas2", 505, 0, 600, 400);
   ntuple->Draw("TMath::Sqrt(px*px + py*py + pz*pz)>>histo(100, 0, 15)", cut);

   // Must be last
   ntuple->Draw("px:py:pz:i", cut, "goff");
}
