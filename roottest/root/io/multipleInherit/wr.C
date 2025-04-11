{
#ifndef SECOND_RUN
      gROOT->ProcessLine(".L na.cxx+");
#endif
      
#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
      gROOT->ProcessLine(".x wr.C");
#else

   TCanvas *c1 = new TCanvas("c1", "c1");
   gStyle->SetPalette(1,0);

   //TFile f("hsimple.root");
   TH1 * h = new TH1F("hpxpy","fake hpxpy",10,0,10); // f.Get("hpxpy");
   //h->SetDirectory(gROOT);
   ////f.Close();
   h->Draw("colz");
   c1->Modified(kTRUE);
   c1->Update();
//
   TMrbNamedArrayI * na = new TMrbNamedArrayI("xyz", "yyy");
   Int_t ind[2] = {3, 5};
   na->Set(2, ind);
   h->GetListOfFunctions()->Add(na);
   //h->GetListOfFunctions()->Print();

#ifdef ClingWorkAroundIncorrectTearDownOrder
  TFile *f = TFile::Open("hout.root","recreate");
#else
  TFile f("hout.root","recreate");
#endif
  h->Write();
   //f.Close();
#endif // SECOND_RUN

#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
