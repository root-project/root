/// \file
/// \ingroup tutorial_multicore
/// \notebook -nodraw
/// Lambdas used to check and fit the result of the H1 analysis.
/// Used by mp104_processH1.C, mp105_processEntryList.C and roottest/root/multicore/tProcessExecutorH1Test.C
///
/// \macro_code
///
/// \author Gerardo Ganis

// This function is used to check the result of the H1 analysis
auto checkH1 = [](TList *out) {

   // Make sure the output list is there
   if (!out) {
      std::cout << "checkH1 >>> Test failure: output list not found\n";
      return -1;
   }

   // Check the 'hdmd' histo
   auto hdmd = dynamic_cast<TH1F *>(out->FindObject("hdmd"));
   if (!hdmd) {
      std::cout << "checkH1 >>> Test failure: 'hdmd' histo not found\n";
      return -1;
   }
   if ((Int_t)(hdmd->GetEntries()) != 7525) {
      std::cout << "checkH1 >>> Test failure: 'hdmd' histo: wrong number"
                   " of entries ("
                << (Int_t)(hdmd->GetEntries()) << ": expected 7525) \n";
      return -1;
   }
   if (TMath::Abs((hdmd->GetMean() - 0.15512023) / 0.15512023) > 0.001) {
      std::cout << "checkH1 >>> Test failure: 'hdmd' histo: wrong mean (" << hdmd->GetMean()
                << ": expected 0.15512023) \n";
      return -1;
   }

   auto h2 = dynamic_cast<TH2F *>(out->FindObject("h2"));
   if (!h2) {
      std::cout << "checkH1 >>> Test failure: 'h2' histo not found\n";
      return -1;
   }
   if ((Int_t)(h2->GetEntries()) != 7525) {
      std::cout << "checkH1 >>> Test failure: 'h2' histo: wrong number"
                   " of entries ("
                << (Int_t)(h2->GetEntries()) << ": expected 7525) \n";
      return -1;
   }
   if (TMath::Abs((h2->GetMean() - 0.15245688) / 0.15245688) > 0.001) {
      std::cout << "checkH1 >>> Test failure: 'h2' histo: wrong mean (" << h2->GetMean() << ": expected 0.15245688) \n";
      return -1;
   }

   // Done
   return 0;
};

// This function is used to fit the result of the analysis with graphics
auto doFit = [](TList *out, const char *lfn = 0) -> Int_t {

   RedirectHandle_t redH;
   if (lfn)
      gSystem->RedirectOutput(lfn, "a", &redH);

   auto hdmd = dynamic_cast<TH1F *>(out->FindObject("hdmd"));
   auto h2 = dynamic_cast<TH2F *>(out->FindObject("h2"));

   // function called at the end of the event loop
   if (hdmd == 0 || h2 == 0) {
      std::cout << "doFit: hdmd = " << hdmd << " , h2 = " << h2 << "\n";
      return -1;
      if (lfn)
         gSystem->RedirectOutput(0, 0, &redH);
   }

   // create the canvas for the h1analysis fit
   gStyle->SetOptFit();
   TCanvas *c1 = new TCanvas("c1", "h1analysis analysis", 10, 10, 800, 600);
   c1->SetBottomMargin(0.15);
   hdmd->GetXaxis()->SetTitle("m_{K#pi#pi} - m_{K#pi}[GeV/c^{2}]");
   hdmd->GetXaxis()->SetTitleOffset(1.4);

   // fit histogram hdmd with function f5 using the log-likelihood option
   if (gROOT->GetListOfFunctions()->FindObject("f5"))
      delete gROOT->GetFunction("f5");

   auto fdm5 = [](Double_t *xx, Double_t *par) -> Double_t {
      const Double_t dxbin = (0.17 - 0.13) / 40; // Bin-width
      Double_t x = xx[0];
      if (x <= 0.13957)
         return 0;
      Double_t xp3 = (x - par[3]) * (x - par[3]);
      Double_t res = dxbin * (par[0] * TMath::Power(x - 0.13957, par[1]) +
                              par[2] / 2.5066 / par[4] * TMath::Exp(-xp3 / 2 / par[4] / par[4]));
      return res;
   };

   auto f5 = new TF1("f5", fdm5, 0.139, 0.17, 5);
   f5->SetParameters(1000000, .25, 2000, .1454, .001);
   hdmd->Fit("f5", "lr");

   // Check the result of the fit
   Double_t ref_f5[4] = {959915.0, 0.351114, 1185.03, 0.145569};
   for (int i : {0, 1, 2, 3}) {
      if ((TMath::Abs((f5->GetParameters())[i] - ref_f5[i]) / ref_f5[i]) > 0.001) {
         std::cout << "\n >>> Test failure: fit to 'f5': parameter '" << f5->GetParName(i) << "' has wrong value ("
                   << (f5->GetParameters())[i] << ": expected" << ref_f5[i] << ") \n";
         if (lfn)
            gSystem->RedirectOutput(0, 0, &redH);
         return -1;
      }
   }

   // create the canvas for tau d0
   gStyle->SetOptFit(0);
   gStyle->SetOptStat(1100);
   auto c2 = new TCanvas("c2", "tauD0", 100, 100, 800, 600);
   c2->SetGrid();
   c2->SetBottomMargin(0.15);

   // Project slices of 2-d histogram h2 along X , then fit each slice
   // with function f2 and make a histogram for each fit parameter
   // Note that the generated histograms are added to the list of objects
   // in the current directory.
   if (gROOT->GetListOfFunctions()->FindObject("f2"))
      delete gROOT->GetFunction("f2");

   auto fdm2 = [](Double_t *xx, Double_t *par) -> Double_t {
      const auto dxbin = (0.17 - 0.13) / 40; // Bin-width
      const auto sigma = 0.0012;
      auto x = xx[0];
      if (x <= 0.13957)
         return 0;
      auto xp3 = (x - 0.1454) * (x - 0.1454);
      auto res = dxbin * (par[0] * TMath::Power(x - 0.13957, 0.25) +
                          par[1] / 2.5066 / sigma * TMath::Exp(-xp3 / 2 / sigma / sigma));
      return res;
   };

   auto f2 = new TF1("f2", fdm2, 0.139, 0.17, 2);
   f2->SetParameters(10000, 10);

   // Restrict to three bins in this example
   std::cout << "doFit: restricting fit to two bins only in this example...\n";

   h2->FitSlicesX(f2, 10, 20, 10, "g5 l");

   // Check the result of the fit
   Double_t ref_f2[2] = {52432.2, 105.481};
   for (int i : {0, 1}) {
      if ((TMath::Abs((f2->GetParameters())[i] - ref_f2[i]) / ref_f2[i]) > 0.001) {
         std::cout << "\n >>> Test failure: fit to 'f2': parameter '" << f2->GetParName(i) << "' has wrong value ("
                   << (f2->GetParameters())[i] << ": expected" << ref_f2[i] << ") \n";
         if (lfn)
            gSystem->RedirectOutput(0, 0, &redH);
         return -1;
      }
   }

   auto h2_1 = (TH1D *)gDirectory->Get("h2_1");
   h2_1->GetXaxis()->SetTitle("#tau[ps]");
   h2_1->SetMarkerStyle(21);
   h2_1->Draw();
   c2->Update();
   auto line = new TLine(0, 0, 0, c2->GetUymax());
   line->Draw();

   // Have the number of entries on the first histogram (to cross check when running
   // with entry lists)
   auto psdmd = (TPaveStats *)hdmd->GetListOfFunctions()->FindObject("stats");
   psdmd->SetOptStat(1110);
   c1->Modified();

   if (lfn)
      gSystem->RedirectOutput(0, 0, &redH);

   return 0;
};

// This is the function invoked during the processing of the trees.
auto doH1 = [](TTreeReader &reader) {

   // Histograms
   auto hdmd = new TH1F("hdmd", "Dm_d", 40, 0.13, 0.17);
   auto h2 = new TH2F("h2", "ptD0 vs Dm_d", 30, 0.135, 0.165, 30, -3, 6);

   TTreeReaderValue<Float_t> fPtds_d(reader, "ptds_d");
   TTreeReaderValue<Float_t> fEtads_d(reader, "etads_d");
   TTreeReaderValue<Float_t> fDm_d(reader, "dm_d");
   TTreeReaderValue<Int_t> fIk(reader, "ik");
   TTreeReaderValue<Int_t> fIpi(reader, "ipi");
   TTreeReaderValue<Int_t> fIpis(reader, "ipis");
   TTreeReaderValue<Float_t> fPtd0_d(reader, "ptd0_d");
   TTreeReaderValue<Float_t> fMd0_d(reader, "md0_d");
   TTreeReaderValue<Float_t> fRpd0_t(reader, "rpd0_t");
   TTreeReaderArray<Int_t> fNhitrp(reader, "nhitrp");
   TTreeReaderArray<Float_t> fRstart(reader, "rstart");
   TTreeReaderArray<Float_t> fRend(reader, "rend");
   TTreeReaderArray<Float_t> fNlhk(reader, "nlhk");
   TTreeReaderArray<Float_t> fNlhpi(reader, "nlhpi");
   TTreeReaderValue<Int_t> fNjets(reader, "njets");

   while (reader.Next()) {

      // Return as soon as a bad entry is detected
      if (TMath::Abs(*fMd0_d - 1.8646) >= 0.04)
         continue;
      if (*fPtds_d <= 2.5)
         continue;
      if (TMath::Abs(*fEtads_d) >= 1.5)
         continue;
      (*fIk)--; // original fIk used f77 convention starting at 1
      (*fIpi)--;

      if (fNhitrp.At(*fIk) * fNhitrp.At(*fIpi) <= 1)
         continue;

      if (fRend.At(*fIk) - fRstart.At(*fIk) <= 22)
         continue;
      if (fRend.At(*fIpi) - fRstart.At(*fIpi) <= 22)
         continue;
      if (fNlhk.At(*fIk) <= 0.1)
         continue;
      if (fNlhpi.At(*fIpi) <= 0.1)
         continue;
      (*fIpis)--;
      if (fNlhpi.At(*fIpis) <= 0.1)
         continue;
      if (*fNjets < 1)
         continue;

      // Fill the histograms
      hdmd->Fill(*fDm_d);
      h2->Fill(*fDm_d, *fRpd0_t / 0.029979 * 1.8646 / *fPtd0_d);
   }

   // Return a list
   auto l = new TList;
   l->Add(hdmd);
   l->Add(h2);
   l->SetOwner(kFALSE);

   return l;
};

// This is the function invoked during the processing of the trees to create a TEntryList
auto doH1fillList = [](TTreeReader &reader) {

   // Entry list
   auto elist = new TEntryList("elist", "H1 selection from Cut");

   TTreeReaderValue<Float_t> fPtds_d(reader, "ptds_d");
   TTreeReaderValue<Float_t> fEtads_d(reader, "etads_d");
   TTreeReaderValue<Int_t> fIk(reader, "ik");
   TTreeReaderValue<Int_t> fIpi(reader, "ipi");
   TTreeReaderValue<Int_t> fIpis(reader, "ipis");
   TTreeReaderValue<Float_t> fMd0_d(reader, "md0_d");
   TTreeReaderArray<Int_t> fNhitrp(reader, "nhitrp");
   TTreeReaderArray<Float_t> fRstart(reader, "rstart");
   TTreeReaderArray<Float_t> fRend(reader, "rend");
   TTreeReaderArray<Float_t> fNlhk(reader, "nlhk");
   TTreeReaderArray<Float_t> fNlhpi(reader, "nlhpi");
   TTreeReaderValue<Int_t> fNjets(reader, "njets");

   while (reader.Next()) {

      // Return as soon as a bad entry is detected
      if (TMath::Abs(*fMd0_d - 1.8646) >= 0.04)
         continue;
      if (*fPtds_d <= 2.5)
         continue;
      if (TMath::Abs(*fEtads_d) >= 1.5)
         continue;
      (*fIk)--; // original fIk used f77 convention starting at 1
      (*fIpi)--;

      if (fNhitrp.At(*fIk) * fNhitrp.At(*fIpi) <= 1)
         continue;

      if (fRend.At(*fIk) - fRstart.At(*fIk) <= 22)
         continue;
      if (fRend.At(*fIpi) - fRstart.At(*fIpi) <= 22)
         continue;
      if (fNlhk.At(*fIk) <= 0.1)
         continue;
      if (fNlhpi.At(*fIpi) <= 0.1)
         continue;
      (*fIpis)--;
      if (fNlhpi.At(*fIpis) <= 0.1)
         continue;
      if (*fNjets < 1)
         continue;

      // Fill the entry list
      elist->Enter(reader.GetCurrentEntry(), reader.GetTree());
   }

   return elist;
};

// This is the function invoked during the processing of the trees using a TEntryList
auto doH1useList = [](TTreeReader &reader) {

   // Histograms
   auto hdmd = new TH1F("hdmd", "Dm_d", 40, 0.13, 0.17);
   auto h2 = new TH2F("h2", "ptD0 vs Dm_d", 30, 0.135, 0.165, 30, -3, 6);

   TTreeReaderValue<Float_t> fDm_d(reader, "dm_d");
   TTreeReaderValue<Float_t> fPtd0_d(reader, "ptd0_d");
   TTreeReaderValue<Float_t> fRpd0_t(reader, "rpd0_t");

   while (reader.Next()) {
      // Fill the histograms
      hdmd->Fill(*fDm_d);
      h2->Fill(*fDm_d, *fRpd0_t / 0.029979 * 1.8646 / *fPtd0_d);
   }

   // Return a list
   auto l = new TList;
   l->Add(hdmd);
   l->Add(h2);
   l->SetOwner(kFALSE);

   return l;
};
