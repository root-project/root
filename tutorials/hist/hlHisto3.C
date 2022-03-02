/// \file
/// \ingroup tutorial_hist
///
/// This tutorial demonstrates how the highlight mechanism can be used on a ntuple.
/// The ntuple in `hsimple.root` is drawn with three differents selection. Moving
/// the mouse ove the two 1D representation display the on 2D plot the events
/// contributing to the highlighted bin.
///
/// \macro_code
///
/// \date March 2018
/// \author Jan Musinsky

TList *list1 = nullptr, *list2 = nullptr;

TList *InitGraphs(TNtuple *nt, TH1F *histo, Bool_t is_top)
{
   Long64_t nev = nt->GetSelectedRows();
   Double_t *px = nt->GetV1();
   Double_t *py = nt->GetV2();
   Double_t *pz = nt->GetV3();

   auto list = new TList();

   Int_t nbins = histo->GetNbinsX();
   for (Int_t bin = 0; bin < nbins; bin++) {
      auto g = new TGraph();
      g->SetName(TString::Format("g%sbin_%d", histo->GetName(), bin+1));
      g->SetBit(kCannotPick);
      g->SetMarkerStyle(25);
      g->SetMarkerColor(histo->GetLineColor());
      list->Add(g);
   }

   for (Long64_t ie = 0; ie < nev; ie++) {
      Double_t value = is_top ? pz[ie] : px[ie]*py[ie]*pz[ie];
      Int_t bin = histo->FindBin(value) - 1;
      auto g = dynamic_cast<TGraph *>(list->At(bin));
      if (g) g->SetPoint(g->GetN(), py[ie], px[ie]);
   }

   return list;
}

void Highlight3(TVirtualPad *pad, TObject *obj, Int_t xhb, Int_t yhb)
{
   auto histo = dynamic_cast<TH1F *>(obj);
   if(!histo) return;

   TCanvas *Canvas2 = (TCanvas *)gROOT->GetListOfCanvases()->FindObject("Canvas2");
   if (!Canvas2) return;
   TGraph *gcommon = (TGraph *)Canvas2->FindObject("Graph");
   if (!gcommon) return;

   TList *list = nullptr;
   if (!strcmp(histo->GetName(), "histo1"))
      list = list1;
   else if (!strcmp(histo->GetName(), "histo2"))
      list = list2;
   if (!list) return;
   TGraph *g = (TGraph *)list->At(xhb);
   if (!g) return;

   TVirtualPad *savepad = gPad;
   Canvas2->cd();
   gcommon->Draw("AP");
   //gcommon->SetTitle(TString::Format("%d / %d", g->GetN(), gcommon->GetN()));
   if (histo->IsHighlight()) // don't draw g after highlight disabled
      if (g->GetN() > 0)
         g->Draw("P");
   Canvas2->Update();
   if (savepad) savepad->cd();
}

void hlHisto3()
{
   auto dir = gROOT->GetTutorialDir();
   dir.Append("/hsimple.C");
   dir.ReplaceAll("/./","/");
   if (!gInterpreter->IsLoaded(dir.Data())) gInterpreter->LoadMacro(dir.Data());
   auto file = (TFile*)gROOT->ProcessLineFast("hsimple(1)");
   if (!file) return;

   TNtuple *ntuple;
   file->GetObject("ntuple", ntuple);
   if (!ntuple) return;
   const char *cut = "pz > 3.0";

   TCanvas *Canvas1 = new TCanvas("Canvas1", "Canvas1", 0, 0, 700, 500);
   Canvas1->HighlightConnect("Highlight3(TVirtualPad*,TObject*,Int_t,Int_t)");
   Canvas1->Divide(1, 2);

   TCanvas *Canvas2 = new TCanvas("Canvas2", "Canvas2", 705, 0, 500, 500);

   // Case1, histo1, pz distribution
   Canvas1->cd(1);
   ntuple->Draw("pz>>histo1(100, 2.0, 12.0)", cut);
   auto histo1 = (TH1F *)gPad->FindObject("histo1");
   auto info1  = new TText(7.0, histo1->GetMaximum()*0.6,
                            "please move the mouse over the frame");
   info1->SetTextColor(histo1->GetLineColor());
   info1->SetBit(kCannotPick);
   info1->Draw();

   // Case2, histo2, px*py*pz distribution
   Canvas1->cd(2);
   ntuple->Draw("(px*py*pz)>>histo2(100, -50.0, 50.0)", cut);
   auto histo2 = (TH1F *)gPad->FindObject("histo2");
   histo2->SetLineColor(kGreen+2);
   auto info2 = new TText(10.0, histo2->GetMaximum()*0.6, info1->GetTitle());
   info2->SetTextColor(histo2->GetLineColor());
   info2->SetBit(kCannotPick);
   info2->Draw();
   Canvas1->Update();

   // set highlight after first drawing completed
   histo1->SetHighlight();
   histo2->SetHighlight();

   // Common graph (all entries, all histo bins)
   Canvas2->cd();
   ntuple->Draw("px:py", cut);
   auto gcommon = (TGraph *)gPad->FindObject("Graph");
   gcommon->SetBit(kCanDelete, kFALSE); // will be redraw
   auto htemp = (TH2F *)gPad->FindObject("htemp");
   gcommon->SetTitle(htemp->GetTitle());
   gcommon->GetXaxis()->SetTitle(htemp->GetXaxis()->GetTitle());
   gcommon->GetYaxis()->SetTitle(htemp->GetYaxis()->GetTitle());
   gcommon->Draw("AP");

   // Must be last
   ntuple->Draw("px:py:pz", cut, "goff");
   list1 = InitGraphs(ntuple, histo1, true);
   list2 = InitGraphs(ntuple, histo2, false);
}
