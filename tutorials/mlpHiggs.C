void mlpHiggs(Int_t ntrain=100) {
// For a LEP search for invisible Higgs boson, a neural network 
// was used to separate the signal from the background passing 
// some selection cuts. Here is a simplified version of this network, 
// taking into account only WW events.
//    Author: Christophe Delaere
   
   if (!gROOT->GetClass("TMultiLayerPerceptron")) {
      gSystem->Load("libMLP");
   }

   // Prepare inputs
   // The 2 trees are merged into one, and a "type" branch, 
   // equal to 1 for the signal and 0 for the background is added.
   TFile input("mlpHiggs.root");
   TTree *signal = (TTree *) input.Get("sig_filtered");
   TTree *background = (TTree *) input.Get("bg_filtered");
   TTree *simu = new TTree("MonteCarlo", "Filtered Monte Carlo Events");
   Float_t ptsumf, qelep, nch, msumf, minvis, acopl, acolin;
   Int_t type;
   signal->SetBranchAddress("ptsumf", &ptsumf);
   signal->SetBranchAddress("qelep",  &qelep);
   signal->SetBranchAddress("nch",    &nch);
   signal->SetBranchAddress("msumf",  &msumf);
   signal->SetBranchAddress("minvis", &minvis);
   signal->SetBranchAddress("acopl",  &acopl);
   signal->SetBranchAddress("acolin", &acolin);
   background->SetBranchAddress("ptsumf", &ptsumf);
   background->SetBranchAddress("qelep",  &qelep);
   background->SetBranchAddress("nch",    &nch);
   background->SetBranchAddress("msumf",  &msumf);
   background->SetBranchAddress("minvis", &minvis);
   background->SetBranchAddress("acopl",  &acopl);
   background->SetBranchAddress("acolin", &acolin);
   simu->Branch("ptsumf", &ptsumf, "ptsumf/F");
   simu->Branch("qelep",  &qelep,  "qelep/F");
   simu->Branch("nch",    &nch,    "nch/F");
   simu->Branch("msumf",  &msumf,  "msumf/F");
   simu->Branch("minvis", &minvis, "minvis/F");
   simu->Branch("acopl",  &acopl,  "acopl/F");
   simu->Branch("acolin", &acolin, "acolin/F");
   simu->Branch("type",   &type,   "type/I");
   type = 1;
   Int_t i;
   for (i = 0; i < signal->GetEntries(); i++) {
      signal->GetEntry(i);
      simu->Fill();
   }
   type = 0;
   for (i = 0; i < background->GetEntries(); i++) {
      background->GetEntry(i);
      simu->Fill();
   }
   // Prepare event lists
   TEventList train;
   TEventList test;
   for (i = 0; i < simu->GetEntries(); i++) {
      if (i % 2)
         train.Enter(i);
      else
         test.Enter(i);
   }
   train.Print();
   test.Print();
   // Build and train the NN
   TMultiLayerPerceptron *mlp = new TMultiLayerPerceptron("msumf/F,ptsumf/F,acolin/F,acopl/F:8:type/I",simu);
   mlp->SetTrainingDataSet(&train);
   mlp->SetTestDataSet(&test);
   mlp->Train(ntrain, "text,graph,update=10");
   // Use the NN to plot the results for each sample
   TH1F *bg = new TH1F("bgh", "NN output", 50, -.5, 1.5);
   TH1F *sig = new TH1F("sigh", "NN output", 50, -.5, 1.5);
   bg->SetDirectory(0);
   sig->SetDirectory(0);
   for (i = 0; i < background->GetEntries(); i++) {
      background->GetEntry(i);
      bg->Fill(mlp->Evaluate(0, (Double_t)msumf, (Double_t)ptsumf, (Double_t)acolin, (Double_t)acopl));
   }
   for (i = 0; i < signal->GetEntries(); i++) {
      signal->GetEntry(i);
      sig->Fill(mlp->Evaluate(0, (Double_t)msumf, (Double_t)ptsumf, (Double_t)acolin, (Double_t)acopl));
   }
   TCanvas *cv = new TCanvas("NNout_cv", "Neural net output");
   bg->SetLineColor(kBlue);
   bg->SetFillStyle(3008);   bg->SetFillColor(kBlue);
   sig->SetLineColor(kRed);
   sig->SetFillStyle(3003); sig->SetFillColor(kRed);
   bg->SetStats(0);
   sig->SetStats(0);
   bg->Draw();
   sig->Draw("same");
   TLegend *legend = new TLegend(.75, .80, .95, .95);
   legend->AddEntry(bg, "Background (WW)");
   legend->AddEntry(sig, "Signal (Higgs)");
   legend->Draw();
}
